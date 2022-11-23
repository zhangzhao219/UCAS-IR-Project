# UCAS-IR-Project

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
