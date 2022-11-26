# UCAS-IR-Project

Information Retrieval Project in UCAS

## 任务描述

使用采样后的TREC 2019训练数据，在TREC 2020 Passage Ranking的子赛道Passage Re-ranking进行检索竞赛。

实现了三种模型，使用七种预训练进行实验

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

**最终的三个模型的实验结果NDCG@10分别为0.7155，0.6739和0.7615**

## 实验日志

Tesla V100 32G * 4，epoch=20，lr=3e-05，seed=42

格式：

时间戳

不训练直接推理2019数据

训练2019数据得到的最好分数

用最好分数对应的模型推理2020数据

|                                             | albert-base-v2                                                    | bert-base-uncased                                                  | electra-base-discriminator                                                  | simlm-base-msmarco                                                | cocodr-base-msmarco                                                | cotmae_base_msmarco_reranker                                       | simlm-msmarco-reranker                                             |
| ------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| bert_sequence_classification<br />batch=256 | 2022_11_20_10_48_17<br />0.1352<br />Epoch =2 0.6837<br />0.6658 | 2022_11_20_10_50_23<br />0.0898<br />Epoch =11 0.6793<br />0.6214 | 2022_11_20_10_54_20<br />0.0404<br />Epoch =19 0.6907<br />0.7067          | 2022_11_20_10_59_04<br />0.0739<br />Epoch =5 0.6597<br />0.6672 | 2022_11_20_11_01_05<br />0.0517<br />Epoch =19 0.6785<br />0.6432 | 2022_11_20_11_02_51<br />0.7296<br />Epoch=1 0.7413<br />0.7605   | 2022_11_20_11_04_03<br />0.7149<br />Epoch =1 0.7192<br />0.7472  |
| bert_cat<br />batch=256                     | 2022_11_20_11_46_20<br />0.0517<br />Epoch=4 0.6888<br />0.6482  | 2022_11_20_11_46_18<br />0.0399<br />Epoch =18 0.6753<br />0.6209 | 2022_11_20_11_47_09<br />0.0752<br />Epoch =4 0.6841<br />**0.7155** | 2022_11_20_11_52_12<br />0.0708<br />Epoch=3 0.6647<br />0.6523  | 2022_11_20_11_53_31<br />0.0408<br />Epoch =6 0.6672<br />0.6443  | 2022_11_20_11_55_30<br />0.3234<br />Epoch=2 0.7361<br />0.7368   | 2022_11_20_11_55_25<br />0.4987<br />Epoch=1 0.72<br />0.7368     |
| colbert<br />batch=128                      | 2022_11_20_13_18_14<br />0.1372<br />Epoch =7 0.4659<br />0.3932 | 2022_11_20_13_18_53<br />0.1654<br />Epoch =17 0.5845<br />0.5508 | 2022_11_20_13_20_26<br />0.0925<br />Epoch=8 0.4708<br />0.43              | 2022_11_20_13_26_46<br />0.1989<br />Epoch=19 0.6066<br />0.5997 | 2022_11_20_13_27_06<br />0.3388<br />Epoch =1 0.69<br />0.6699    | 2022_11_20_13_34_50<br />0.0763<br />Epoch =12 0.6039<br />0.6237 | 2022_11_20_13_34_54<br />0.0787<br />Epoch =20 0.2621<br />0.2427 |

添加warmup学习率预热策略：

|                                             | albert-base-v2                                         | bert-base-uncased                                     | electra-base-discriminator                            | simlm-base-msmarco                                              | cocodr-base-msmarco                                   | cotmae_base_msmarco_reranker                                   | simlm-msmarco-reranker                                 |
| ------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------ |
| bert_sequence_classification<br />batch=256 | 2022_11_21_04_53_02<br />Epoch=4 0.6738<br />0.663    | 2022_11_21_04_54_54<br />Epoch=5 0.657<br />0.6368   | 2022_11_21_04_58_55<br />Epoch =6 0.6819<br />0.6911 | 2022_11_21_05_01_28<br />Epoch =4 0.6555<br />**0.6739** | 2022_11_21_05_03_43<br />Epoch =3 0.6726<br />0.6537 | 2022_11_21_05_05_06<br />Epoch=1 0.7327<br />**0.7615** | 2022_11_21_05_00_42<br />Epoch=2 0.7157<br />0.7456   |
| bert_cat<br />batch=256                     | 2022_11_21_05_45_13<br />Epoch =2 0.6781<br />0.6244  | 2022_11_21_05_45_05<br />Epoch=8 0.6823<br />0.6215  | 2022_11_21_05_50_20<br />Epoch =4 0.6819<br />0.7035 | 2022_11_21_05_50_45<br />Epoch =3 0.6595<br />0.6553           | 2022_11_21_05_53_00<br />Epoch =8 0.6685<br />0.6423 | 2022_11_21_05_55_13<br />Epoch=2 0.7409<br />0.755            | 2022_11_21_05_50_13<br />Epoch =2 0.6977<br />0.7222  |
| colbert<br />batch=128                      | 2022_11_21_06_37_14<br />Epoch =12 0.5467<br />0.4688 | 2022_11_21_06_35_51<br />Epoch=18 0.5839<br />0.5163 | 2022_11_21_06_41_48<br />Epoch=16 0.4515<br />0.355  | 2022_11_21_08_27_35<br />Epoch =16 0.6035<br />0.6049          | 2022_11_21_06_42_38<br />Epoch=5 0.6906<br />0.6578  | 2022_11_21_06_45_33<br />Epoch =16 0.5712<br />0.5803         | 2022_11_21_06_39_30<br />Epoch =14 0.1903<br />0.1766 |

## 训练测试和预测流程

注：由于文件中包含了原始的预训练模型、训练好的模型和镜像，附件比较大。

如果失效，请至[Google Drive](https://drive.google.com/drive/folders/1jEgrCuCsCVIS1ACE_Brwo0mOm4L1dWNe?usp=sharing)下载

解压zip文件至本地目录中

```bash
unzip zhangzhao-IR.zip -d zhangzhao-IR
```

提供了Docker和python两种方法运行程序，推荐使用Docker

如果使用docker，将 `IR.tar.gz`解压到 `zhangzhao-IR/images`目录下：

```bash
tar xvzf IR.tar.gz -C zhangzhao-IR/images
```

### 文件说明

切换到 `zhangzhao-IR`目录下，有如下文件：

```bash
.
├── README_submit.md # 说明文件
├── file
│   ├── bert_cat.py # 模型文件1
│   ├── bert_sequence_classification.py # 模型文件2
│   ├── colbert.py # 模型文件3
│   ├── data # 原始数据
│   │   ├── 2019 # 2019年原始数据
│   │   │   ├── 2019qrels-pass.txt
│   │   │   ├── collection.train.sampled.tsv
│   │   │   ├── msmarco-passagetest2019-43-top1000.tsv
│   │   │   ├── qidpidtriples.train.sampled.tsv
│   │   │   └── queries.train.sampled.tsv
│   │   └── 2020 # 2020年原始数据
│   │       ├── 2020qrels-pass.txt
│   │       └── msmarco-passagetest2020-54-top1000.tsv
│   ├── dataset.py # 数据处理文件
│   ├── ensemble_rank.py # 模型集成文件（未使用）
│   ├── ensemble_score.py # 模型集成文件（未使用）
│   ├── log.py # 日志配置文件
│   ├── main.py # 主文件
│   ├── models # 训练好的模型文件
│   │   ├── cotmae_base_msmarco_reranker-bert_sequence_classification
│   │   │   └── best.pt
│   │   ├── electra-base-discriminator-bert_cat
│   │   │   └── best.pt
│   │   └── simlm-base-msmarco-bert_sequence_classification
│   │       └── best.pt
│   ├── only_predict # 仅预测脚本
│   │   ├── run_cotmae_base_msmarco_reranker-bert_sequence_classification.sh
│   │   ├── run_electra-base-discriminator-bert_cat.sh
│   │   └── run_simlm-base-msmarco-bert_sequence_classification.sh
│   ├── pretrained # 预训练模型文件
│   │   ├── caskcsg
│   │   │   └── cotmae_base_msmarco_reranker
│   │   │       ├── config.json
│   │   │       ├── pytorch_model.bin
│   │   │       ├── special_tokens_map.json
│   │   │       ├── tokenizer_config.json
│   │   │       └── vocab.txt
│   │   ├── google
│   │   │   └── electra-base-discriminator
│   │   │       ├── config.json
│   │   │       ├── pytorch_model.bin
│   │   │       ├── tokenizer.json
│   │   │       ├── tokenizer_config.json
│   │   │       └── vocab.txt
│   │   └── intfloat
│   │       └── simlm-base-msmarco
│   │           ├── config.json
│   │           ├── pytorch_model.bin
│   │           ├── special_tokens_map.json
│   │           ├── tokenizer.json
│   │           ├── tokenizer_config.json
│   │           └── vocab.txt
│   ├── requirements.txt # 依赖库文件
│   ├── result # 运行结果
│   │   ├── log_2022_11_20_11_47_09 # 运行结果日志
│   │   ├── log_2022_11_21_05_01_28 # 运行结果日志
│   │   ├── log_2022_11_21_05_05_06 # 运行结果日志
│   │   ├── result_2019qrels_2022_11_20_11_47_09 # 在2019年数据上进行验证的TREC结果文件
│   │   ├── result_2019qrels_2022_11_21_05_01_28 # 在2019年数据上进行验证的TREC结果文件
│   │   ├── result_2019qrels_2022_11_21_05_05_06 # 在2019年数据上进行验证的TREC结果文件
│   │   ├── result_2020qrels_2022_11_20_11_47_09 # 在2020年数据上进行推理后的TREC结果文件
│   │   ├── result_2020qrels_2022_11_21_05_01_28 # 在2020年数据上进行推理后的TREC结果文件
│   │   └── result_2020qrels_2022_11_21_05_05_06 # 在2020年数据上进行推理后的TREC结果文件
│   ├── rich_progress.py # 命令行美化配置
│   ├── train_and_predict # 训练+训练后预测脚本
│   │   ├── run_cotmae_base_msmarco_reranker-bert_sequence_classification.sh
│   │   ├── run_electra-base-discriminator-bert_cat.sh
│   │   └── run_simlm-base-msmarco-bert_sequence_classification.sh
│   └── trec_eval-9.0.7.tar.gz # 评测脚本
├── images # Docker镜像路径
│   └── IR.tar # Docker镜像
├── install_docker.sh # Docker安装脚本
├── install_python.sh # Python依赖库安装脚本
├── predict_docker.sh # Docker仅预测脚本
├── predict_python.sh # Python仅预测脚本
├── train_and_predict_docker.sh # Docker训练+预测脚本
├── train_and_predict_python.sh # Python训练+预测脚本
```

### Docker训练测试和预测流程

**注：需要安装docker并且有权限（sudo）**

#### 安装镜像和评测脚本

```bash
sudo bash install_docker.sh
```

#### 一键推理

```bash
sudo bash predict_docker.sh
```

会调用 `file/models`内部的3个模型进行分别推理，输出的结果文件在 `file/`中，名称分别为

```bash
result_2020qrels_electra-base-discriminator-bert_cat
result_2020qrels_simlm-base-msmarco-bert_sequence_classification
result_2020qrels_cotmae_base_msmarco_reranker-bert_sequence_classification
```

**由于推理使用的机器可能不同，结果可能有一点点差异**

提交版本是 `NVIDIA Tesla V100 32G * 1`上进行推理后得到的结果

#### 一键训练+推理

```bash
sudo bash train_and_predict_docker.sh
```

训练注意事项：

1. 如果GPU的数量足够，上述脚本内部调用的3个脚本可以并行运行，更改GPU卡号在每个文件的第三行 `GPU='0 1 2 3'`
2. 训练过程是在 `NVIDIA Tesla V100 32G * 4`上进行的，**更换硬件或者缩小batch_size对最终结果都会有一定影响**
3. 训练后的模型存放在 `file/models/run 1/2/3`中，训练后推理也会调用这些模型进行推理
4. 有输出日志，存放在 `file/log_run 1/2/3`文件中

### Python训练测试和预测流程

**注：需要创建一个Python版本为3.8.13的环境并 `conda activate 环境名称`**

#### 安装依赖包和评测脚本

```bash
bash install_python.sh
```

#### 一键推理

```bash
bash predict_python.sh
```

会调用 `file/models`内部的3个模型进行分别推理，输出的结果文件在 `file/`中，名称分别为

```bash
result_2020qrels_electra-base-discriminator-bert_cat
result_2020qrels_simlm-base-msmarco-bert_sequence_classification
result_2020qrels_cotmae_base_msmarco_reranker-bert_sequence_classification
```

**由于推理使用的机器可能不同，结果可能有非常微小的差异**

提交版本是 `NVIDIA Tesla V100 32G * 1`上进行推理后得到的结果

#### 一键训练+推理

```bash
bash train_and_predict_python.sh
```

训练注意事项：

1. 如果GPU的数量足够，上述脚本内部调用的3个脚本可以并行运行，更改GPU卡号在每个文件的第三行 `GPU='0 1 2 3'`
2. 训练过程是在 `NVIDIA Tesla V100 32G * 4`上进行的，**更换硬件或者缩小batch_size对最终结果都会有一定影响**
3. 训练后的模型存放在 `file/models/run 1/2/3`中，训练后推理也会调用这些模型进行推理
4. 有输出日志，存放在 `file/log_run 1/2/3`文件中
