# Variables
SEED=42
GPU='0 1 2 3'
TEST_BATCH=1024
TRAIN_BATCH=256

# MODEL=bert_sequence_classification
# MODEL=bert_cat
# MODEL=colbert

# BERT='pretrained/albert-base-v2'
# BERT='pretrained/bert-base-uncased'
# BERT='pretrained/google/electra-base-discriminator'

# BERT='pretrained/intfloat/simlm-base-msmarco'
# BERT='pretrained/OpenMatch/cocodr-base-msmarco'

# BERT='pretrained/caskcsg/cotmae_base_msmarco_reranker'
# BERT='pretrained/intfloat/simlm-msmarco-reranker'

TIMESTAMP=$(date +%Y_%m_%d_%H_%M_%S)

echo $TIMESTAMP

nvidia-smi

# tar -xvf trec_eval-9.0.7.tar.gz
# cd trec_eval-9.0.7
# make
# cd ..

# 测试2019年的数据
python main.py \
--test \
--batch ${TEST_BATCH}  --datetime ${TIMESTAMP} --gpu ${GPU} \
--data_folder_dir 2019 --test_data_file msmarco-passagetest2019-43-top1000.tsv --test_result_file 2019qrels-pass.txt \
--bert ${BERT} --model ${MODEL}

# 使用2019年的数据进行训练
python main.py \
--train \
--batch ${TRAIN_BATCH} --board --datetime ${TIMESTAMP} --epoch 20 --gpu ${GPU} --lr 3e-5 --seed 42 --early_stop 20 \
--data_folder_dir 2019 --train_document collection.train.sampled.tsv --train_query queries.train.sampled.tsv --qid_pid qidpidtriples.train.sampled.tsv --test_data_file msmarco-passagetest2019-43-top1000.tsv --test_result_file 2019qrels-pass.txt \
--save --bert ${BERT} --model ${MODEL}

# 测试2019年的数据
python main.py \
--test \
--batch ${TEST_BATCH}  --datetime ${TIMESTAMP} --gpu ${GPU} \
--data_folder_dir 2019 --test_data_file msmarco-passagetest2019-43-top1000.tsv --test_result_file 2019qrels-pass.txt \
--load --bert ${BERT} --model ${MODEL}

# 测试2020年的数据
python main.py \
--predict \
--batch ${TEST_BATCH}  --datetime ${TIMESTAMP} --gpu ${GPU} \
--data_folder_dir 2020 --test_data_file msmarco-passagetest2020-54-top1000.tsv --test_result_file 2020qrels-pass.txt \
--load --bert ${BERT} --model ${MODEL}