directory=`pwd`

docker run -v ${directory}/file:/mnt --gpus all --shm-size=6g -it zhangzhao-ir:v1 /bin/bash -c 'cd /mnt && bash only_predict/run_electra-base-discriminator-bert_cat.sh'

docker run -v ${directory}/file:/mnt --gpus all --shm-size=6g -it zhangzhao-ir:v1 /bin/bash -c 'cd /mnt && bash only_predict/run_simlm-base-msmarco-bert_sequence_classification.sh'

docker run -v ${directory}/file:/mnt --gpus all --shm-size=6g -it zhangzhao-ir:v1 /bin/bash -c 'cd /mnt && bash only_predict/run_cotmae_base_msmarco_reranker-bert_sequence_classification.sh'
