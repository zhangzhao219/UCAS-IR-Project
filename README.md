# UCAS-IR-Project

Information Retrieval Project in UCAS

目前实现了三种模型，使用了七种预训练正在进行实验

在前期实验中：

2019年NDCG@10可以达到0.719（第三类），0.6607（第二类）

2020年NDCG@10可以达到0.7511（第三类），0.6691（第二类）

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

更改`run_example.sh`即可运行代码（记得调整参数）
