import os
import sys
import logging
import numpy as np
import pandas as pd
from log import config_logging
from itertools import combinations

SIGNSUM = 'rank'
ASCENDING = True

# log
config_logging("log_ensemble_rank")
logging.info('Log is ready!')

DATA_FOLDER = 'goodresults'

data_list = []
data_name = []
for file in os.listdir(DATA_FOLDER):
    data = pd.read_csv(os.path.join(DATA_FOLDER,file),sep='\t')
    data.columns = ['qid','Q0','did','rank','score','des']
    data = data.sort_values(by=['qid','did'])
    data_list.append(data)
    data_name.append(file)

score1 = []
score2 = []
score3 = []
score = []

for num in range(2, len(data_list)+1):
    temp_data = list(combinations(data_list, num))
    temp_name = list(combinations(data_name, num))
    for i in range(0,len(temp_data)):
        logging.info(temp_name[i])

        data_output = temp_data[i][0].copy()
        for j in range(1,len(temp_data[i])):
            data_output[SIGNSUM] += temp_data[i][j][SIGNSUM]

        data_output = data_output.sort_values(by=['qid',SIGNSUM], ascending=ASCENDING)
        for qid in data_output['qid'].unique():
            data_output.loc[data_output['qid'] == qid,'rank'] = [i for i in range(1, len(data_output.loc[data_output['qid'] == qid,'rank'])+1)]
        data_output.to_csv('temp_result_rank', index=None, header=None, sep='\t')

        eva = os.popen('trec_eval-9.0.7/trec_eval -m  ndcg_cut ' + 'data/2020/2020qrels-pass.txt' + ' ' + 'temp_result_rank').readlines()[1].split(' ')[-1].split('\t')[-1].split('\n')[0]
        logging.info(float(eva))
        signlist = [k.split('_')[0] for k in temp_name[i]]
        if len(set(signlist)) != 1:
            score.append(float(eva))
        else:
            if signlist[0] == '1':
                score1.append(float(eva))
            elif signlist[0] == '2':
                score2.append(float(eva))
            elif signlist[0] == '3':
                score3.append(float(eva))

logging.info(score)
logging.info(max(score))
logging.info(score1)
logging.info(max(score1))
logging.info(score2)
logging.info(max(score2))
logging.info(score3)
logging.info(max(score3))