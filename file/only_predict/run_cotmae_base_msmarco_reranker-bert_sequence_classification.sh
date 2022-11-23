# Variables
GPU='0'
TEST_BATCH=256

MODEL=bert_sequence_classification

BERT='pretrained/caskcsg/cotmae_base_msmarco_reranker'

TIMESTAMP=cotmae_base_msmarco_reranker-bert_sequence_classification

echo $TIMESTAMP

nvidia-smi

# 测试2020年的数据
python main.py \
--predict \
--batch ${TEST_BATCH}  --datetime ${TIMESTAMP} --gpu ${GPU} \
--data_folder_dir 2020 --test_data_file msmarco-passagetest2020-54-top1000.tsv --test_result_file 2020qrels-pass.txt \
--load --bert ${BERT} --model ${MODEL}