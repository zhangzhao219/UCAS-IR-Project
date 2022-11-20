import os
import random
import logging
import argparse
import importlib

import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from collections import deque

from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from log import config_logging
from dataset import LoadTrainData, LoadTestData, LoadTrainData_Colbert, LoadTestData_Colbert
from rich_progress import progress

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Pytorch NLP')

parser.add_argument('--train', action='store_true', help='Whether to train')
parser.add_argument('--test', action='store_true', help='Whether to test')
parser.add_argument('--predict', action='store_true', help='Whether to predict')

parser.add_argument('--batch', type=int, default=64, help='Define the batch size')
parser.add_argument('--board', action='store_true', help='Whether to use tensorboard')
parser.add_argument('--datetime', type=str, required=True, help='Get Time Stamp')
parser.add_argument('--epoch', type=int, default=50, help='Training epochs')
parser.add_argument('--gpu', type=str, nargs='+', help='Use GPU')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--seed',type=int, default=42, help='Random Seed')
parser.add_argument('--early_stop',type=int, default=10, help='Early Stop Epoch')

parser.add_argument('--data_folder_dir', type=str, required=True, help='Data Folder Location')

parser.add_argument('--train_document', type=str, help='Train Document Filename')
parser.add_argument('--train_query', type=str, help='Train Query Filename')
parser.add_argument('--qid_pid', type=str, help='Document vs Query Filename')

parser.add_argument('--test_data_file', type=str, help='Test Data Filename')
parser.add_argument('--test_result_file', type=str, help='Test Result Filename')

parser.add_argument('--save', action='store_true', help='Whether to save model')
parser.add_argument('--load', action='store_true', help='Whether to load best model')

parser.add_argument('--bert', type=str, required=True, help='Choose Bert')
parser.add_argument('--model', type=str, required=True, help='Model type')

parser.add_argument('--warmup', type=float, default=0.0, help='warm up ratio')

args = parser.parse_args()

model_structure = importlib.import_module(args.model)
PretrainedModel = model_structure.PretrainedModel
getTokenizer = model_structure.getTokenizer

TIMESTAMP = args.datetime

# log
config_logging("log_" + TIMESTAMP)
logging.info('Log is ready!')
logging.info(args)

if args.board:
    writer = SummaryWriter('runs/' + TIMESTAMP)
    logging.info('Tensorboard is ready!')

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.info('GPU: ' + ','.join(args.gpu))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.gpu:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info('Seed: ' + str(seed)) 

def read_train_data(train_document, train_query, qid_pid):
    document = pd.read_csv(train_document, sep='\t', header=None)
    document.columns = ['key', 'value']
    document_dict = dict(zip(document['key'], document['value']))

    query = pd.read_csv(train_query, sep='\t', header=None)
    query.columns = ['key', 'value']
    query_dict = dict(zip(query['key'], query['value']))

    data = pd.read_csv(qid_pid, sep='\t', header=None)
    data.columns = ['query', 'positive', 'negative']

    data['query'] = data['query'].map(query_dict)
    data['positive'] = data['positive'].map(document_dict)
    data['negative'] = data['negative'].map(document_dict)
    
    logging.info('Read train data')
    return data 

def read_test_data(data_path):
    data = pd.read_csv(data_path,sep='\t',header=None)
    logging.info('Read test data: ' + data_path)
    return data

def train(args, model, data):

    logging.info(f'Start Training!')
    # use GPU
    if args.gpu:
        model = model.cuda()
        if len(args.gpu) >= 2:
            model= nn.DataParallel(model)
    
    logging.info(f'Load {args.bert} Tokenizer')
    tokenizer = getTokenizer(args.bert)
    # data
    if args.model == "colbert":
        train_dataset = LoadTrainData_Colbert(data, tokenizer)
    else:
        train_dataset = LoadTrainData(data, tokenizer)
    # len(data)
    dataset_len = train_dataset.__len__()

    # 由于在bert官方的代码中对于bias项、LayerNorm.bias、LayerNorm.weight项是免于正则化的。因此经常在bert的训练中会采用与bert原训练方式一致的做法
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    criterion = nn.CrossEntropyLoss()
    
    # warmup
    scheduler = optimizer

    if args.warmup != 0.0:
        num_train_optimization_steps = dataset_len / args.baloadtch * args.epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_optimization_steps*args.warmup), num_train_optimization_steps)
        
    early_stop_sign = deque(maxlen=args.early_stop)

    best_metric = 0.0

    # epoch_task = progress.add_task("Epoch...", total=args.epoch)

    # loops
    for epoch in range(args.epoch):
        # dataset loader
        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=False)

        epoch_iterator = tqdm(train_loader, desc="Iteration", total=len(train_loader))
        epoch_iterator.set_description(f'Train-{epoch}')

        # train_task = progress.add_task("Training...", total=len(train_loader))

        # progress.reset(train_task)

        # set train mode
        model.train()

        loss = 0

        for step, input_tuple in enumerate(epoch_iterator):

            if len(input_tuple) == 6:

                input_ids_positive, token_type_ids_positive, attention_mask_positive, input_ids_negative, token_type_ids_negative, attention_mask_negative = input_tuple
            
            else:
                query_input_ids, query_token_type_ids, query_attention_mask, input_ids_positive, token_type_ids_positive, attention_mask_positive, input_ids_negative, token_type_ids_negative, attention_mask_negative = input_tuple

                query_input_ids = query_input_ids.squeeze(1)
                query_token_type_ids = query_token_type_ids.squeeze(1)
                query_attention_mask = query_attention_mask.squeeze(1)

            input_ids_positive = input_ids_positive.squeeze(1)
            token_type_ids_positive = token_type_ids_positive.squeeze(1)
            attention_mask_positive = attention_mask_positive.squeeze(1)
            input_ids_negative = input_ids_negative.squeeze(1)
            token_type_ids_negative = token_type_ids_negative.squeeze(1)
            attention_mask_negative = attention_mask_negative.squeeze(1)

            if args.gpu:
                model = model.cuda()
                input_ids_positive = input_ids_positive.cuda()
                token_type_ids_positive = token_type_ids_positive.cuda()
                attention_mask_positive = attention_mask_positive.cuda()
                input_ids_negative = input_ids_negative.cuda()
                token_type_ids_negative = token_type_ids_negative.cuda()
                attention_mask_negative = attention_mask_negative.cuda()

                if len(input_tuple) == 9:
                    query_input_ids = query_input_ids.cuda()
                    query_token_type_ids = query_token_type_ids.cuda()
                    query_attention_mask = query_attention_mask.cuda()

            if len(input_tuple) == 6:
                output_positive = model(input_ids_positive, token_type_ids_positive, attention_mask_positive)
                output_negative = model(input_ids_negative, token_type_ids_negative, attention_mask_negative)

            else:
                output_positive = model(query_input_ids, query_token_type_ids, query_attention_mask, input_ids_positive, token_type_ids_positive, attention_mask_positive)
                output_negative = model(query_input_ids, query_token_type_ids, query_attention_mask, input_ids_negative, token_type_ids_negative, attention_mask_negative)
            
            # loss_single = torch.mean(-(torch.exp(output_positive.logits) / (torch.exp(output_positive.logits) + torch.exp(output_negative.logits))).log(),dim=0)
            loss_single = -torch.mean(output_positive - torch.log(torch.exp(output_positive) + torch.exp(output_negative) + 1e-7), dim=0)

            loss += loss_single.item()

            # backward 
            loss_single.backward()

            optimizer.step()

            if args.warmup != 0.0:
                scheduler.step()

            model.zero_grad() # zero grad

            # progress.update(train_task, advance=1)

            # renew tqdm
            epoch_iterator.update(1)
            # add description in the end
            epoch_iterator.set_postfix(loss=loss_single.item())

        epoch_iterator.close()

        metric = test(args, model, TEST_DATA_FILE, test_dataset, test_loader)

        logging.info(f'Eval Epoch = {epoch+1} NDCG_10:{metric}')
        
        if metric > best_metric:
            logging.info(f'Test NDCG_10:{metric} > max_metric!')
            best_metric = metric

            if args.save:
                torch.save(model.state_dict(), MODEL_PATH + 'best.pt')
                logging.info(f'Best Model Saved!')
        else:
            early_stop_sign.append(1)
            if sum(early_stop_sign) == args.early_stop:
                logging.info(f'The Effect of last {args.early_stop} epochs has not improved! Early Stop!')
                logging.info(f'Best NDCG_10: {best_metric}')
                break

        # tensorboard
        if args.board:
            writer.add_scalar(f'Loss', loss / args.batch, epoch+1)
            writer.add_scalar(f'NDCG_10', metric, epoch+1)
        
        # progress.update(epoch_task, advance=1)

    if args.gpu:
        torch.cuda.empty_cache()

# test 2019 data
def test(args, model, data, test_dataset, test_loader):
    logging.info('Start evaluate!')

    # store result
    predict_result = np.empty((test_dataset.__len__(),1))

    if args.load:
        if args.gpu:
            model = model.cuda()
            model = nn.DataParallel(model)
        model.load_state_dict(torch.load(MODEL_PATH + 'best.pt'))
        logging.info(f'best.pt Loaded!')

    # set eval mode
    model.eval()

    # test_task = progress.add_task("Testing...", total=len(test_loader))

    epoch_iterator = tqdm(test_loader, desc="Iteration", total=len(test_loader))

    for step, input_tuple in enumerate(test_loader):

        if len(input_tuple) == 6:
            query_input_ids, query_token_type_ids, query_attention_mask, input_ids, token_type_ids, attention_mask = input_tuple
            query_input_ids = query_input_ids.squeeze(1)
            query_token_type_ids = query_token_type_ids.squeeze(1)
            query_attention_mask = query_attention_mask.squeeze(1)
        
        else:
            input_ids, token_type_ids, attention_mask = input_tuple

        input_ids = input_ids.squeeze(1)
        token_type_ids = token_type_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        if args.gpu:
            model = model.cuda()
            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
            attention_mask = attention_mask.cuda()
            if len(input_tuple) == 6:
                query_input_ids = query_input_ids.cuda()
                query_token_type_ids = query_token_type_ids.cuda()
                query_attention_mask = query_attention_mask.cuda()
            
        with torch.no_grad():
            if len(input_tuple) == 6:
                output = model(query_input_ids, query_token_type_ids, query_attention_mask, input_ids, token_type_ids, attention_mask)
            else:
                output = model(input_ids, token_type_ids, attention_mask)

        if args.gpu:
            output = output.cpu()

        predict_result[step*args.batch:step*args.batch + output.shape[0]] = output

        # progress.update(test_task, advance=1)

        # renew tqdm
        epoch_iterator.update(1)

    epoch_iterator.close()

    # evaluate
    ndcg10 = evaluation(args, data, predict_result)
    logging.info(f'NDCG_10: {ndcg10}')

    return ndcg10

# calculate score and store result
def evaluation(args, data, score):

    data['score'] = score
    data.columns = ['qid','did','query','document','score']

    df_empty = pd.DataFrame(columns=['qid', 'Q0', 'did', 'rank','rating','name'])

    df_list = []

    for qid in data.iloc[:,0].unique():
        df_empty_t = df_empty.copy()
        df_empty_t['did'] = data.loc[data['qid'] == qid,'did']
        df_empty_t['rating'] = data.loc[data['qid'] == qid,'score']
        df_empty_t['qid'] = qid
        df_empty_t = df_empty_t.sort_values(by=['rating'], ascending=False)
        df_empty_t['rank'] = [i for i in range(1, len(df_empty_t)+1)]
        df_list.append(df_empty_t)

    df_empty = pd.concat(df_list)
    df_empty['Q0'] = 'Q0'
    df_empty['name'] = args.bert

    result_file_name = 'result_' + args.test_result_file.split('-')[0] + '_' + TIMESTAMP

    df_empty.to_csv(result_file_name, index=None, header=None, sep='\t')

    eva = os.popen('trec_eval-9.0.7/trec_eval -m  ndcg_cut ' + TEST_RESULT_FILE + ' ' + result_file_name).readlines()[1].split(' ')[-1].split('\t')[-1].split('\n')[0]

    return float(eva)

if __name__ == '__main__':

    # progress.start()

    # set seed
    set_seed(args.seed)

    logging.info(f'Load {args.bert} Model')
    model = PretrainedModel(args.bert)

    MODEL_PATH = 'models/' + TIMESTAMP + '/'
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    DATA_PATH = 'data/'
    DATA_FOLDER = os.path.join(DATA_PATH,args.data_folder_dir)
    
    TEST_RESULT_FILE = os.path.join(DATA_FOLDER,args.test_result_file)
    TEST_DATA_FILE = read_test_data(os.path.join(DATA_FOLDER,args.test_data_file))

    # test data
    if args.model == "colbert":
        test_dataset = LoadTestData_Colbert(TEST_DATA_FILE,getTokenizer(args.bert))
    else:
        test_dataset = LoadTestData(TEST_DATA_FILE,getTokenizer(args.bert))
    test_loader = DataLoader(test_dataset,batch_size=args.batch,shuffle=False,drop_last=False)

    if args.train:

        TRAIN_DATA_FILE = read_train_data(os.path.join(DATA_FOLDER,args.train_document),os.path.join(DATA_FOLDER,args.train_query),os.path.join(DATA_FOLDER,args.qid_pid))

        train(args, model, TRAIN_DATA_FILE)

    if args.test or args.predict:
        test(args, model, TEST_DATA_FILE, test_dataset, test_loader)