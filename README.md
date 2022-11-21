# UCAS-IR-Project

Information Retrieval Project in UCAS

目前实现了三种模型，使用了七种预训练正在进行实验

```bash
MODEL=bert_sequence_classification
MODEL=bert_cat
MODEL=colbert
# 普通预训练模型
BERT='pretrained/albert-base-v2'
BERT='pretrained/bert-base-uncased'
BERT='pretrained/google/electra-base-discriminator'
# 在msmarco数据上训练的预训练模型
BERT='pretrained/intfloat/simlm-base-msmarco'
BERT='pretrained/OpenMatch/cocodr-base-msmarco'
# 在msmarco数据上训练，并在Rerank任务上进行微调的预训练模型
BERT='pretrained/caskcsg/cotmae_base_msmarco_reranker'
BERT='pretrained/intfloat/simlm-msmarco-reranker'
```

更改 `run_example.sh`即可运行代码（记得调整参数）

## 实验结果

Tesla V100 32G * 4，epoch=20，lr=3e-05，seed=42

格式：

时间戳

不训练直接推理2019数据

训练2019数据得到的最好分数

用最好分数对应的模型推理2020数据

|                                             | albert-base-v2                                                    | bert-base-uncased                                                  | electra-base-discriminator                                         | simlm-base-msmarco                                                | cocodr-base-msmarco                                                | cotmae_base_msmarco_reranker                                       | simlm-msmarco-reranker                                             |
| ------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ | ----------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| bert_sequence_classification<br />batch=256 | 2022_11_20_10_48_17<br />0.1352<br />Epoch =2 0.6837<br />0.6658 | 2022_11_20_10_50_23<br />0.0898<br />Epoch =11 0.6793<br />0.6214 | 2022_11_20_10_54_20<br />0.0404<br />Epoch =19 0.6907<br />0.7067 | 2022_11_20_10_59_04<br />0.0739<br />Epoch =5 0.6597<br />0.6672 | 2022_11_20_11_01_05<br />0.0517<br />Epoch =19 0.6785<br />0.6432 | 2022_11_20_11_02_51<br />0.7296<br />Epoch=1 0.7413<br />0.7605   | 2022_11_20_11_04_03<br />0.7149<br />Epoch =1 0.7192<br />0.7472  |
| bert_cat<br />batch=256                     | 2022_11_20_11_46_20<br />0.0517<br />Epoch=4 0.6888<br />0.6482  | 2022_11_20_11_46_18<br />0.0399<br />Epoch =18 0.6753<br />0.6209 | 2022_11_20_11_47_09<br />0.0752<br />Epoch =4 0.6841<br />0.7155  | 2022_11_20_11_52_12<br />0.0708<br />Epoch=3 0.6647<br />0.6523  | 2022_11_20_11_53_31<br />0.0408<br />Epoch =6 0.6672<br />0.6443  | 2022_11_20_11_55_30<br />0.3234<br />Epoch=2 0.7361<br />0.7368   | 2022_11_20_11_55_25<br />0.4987<br />Epoch=1 0.72<br />0.7368     |
| colbert<br />batch=128                      | 2022_11_20_13_18_14<br />0.1372<br />Epoch =7 0.4659<br />0.3932 | 2022_11_20_13_18_53<br />0.1654<br />Epoch =17 0.5845<br />0.5508 | 2022_11_20_13_20_26<br />0.0925<br />Epoch=8 0.4708<br />0.43     | 2022_11_20_13_26_46<br />0.1989<br />Epoch=19 0.6066<br />0.5997 | 2022_11_20_13_27_06<br />0.3388<br />Epoch =1 0.69<br />0.6699    | 2022_11_20_13_34_50<br />0.0763<br />Epoch =12 0.6039<br />0.6237 | 2022_11_20_13_34_54<br />0.0787<br />Epoch =20 0.2621<br />0.2427 |

添加warmup学习率预热策略：

|                                             | albert-base-v2      | bert-base-uncased   | electra-base-discriminator | simlm-base-msmarco  | cocodr-base-msmarco | cotmae_base_msmarco_reranker | simlm-msmarco-reranker |
| ------------------------------------------- | ------------------- | ------------------- | -------------------------- | ------------------- | ------------------- | ---------------------------- | ---------------------- |
| bert_sequence_classification<br />batch=256 | 2022_11_21_04_53_02 | 2022_11_21_04_54_54 | 2022_11_21_04_58_55        | 2022_11_21_05_01_28 | 2022_11_21_05_03_43 | 2022_11_21_05_05_06          | 2022_11_21_05_00_42    |
| bert_cat<br />batch=256                     |                     |                     |                            |                     |                     |                              |                        |
| colbert<br />batch=128                      |                     |                     |                            |                     |                     |                              |                        |
