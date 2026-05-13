# LSTM 中文商品评论情感分析

本项目基于 **PyTorch + LSTM + Jieba 分词** 实现中文商品评论情感分析任务。项目使用 `online_shopping_10_cats.csv` 数据集，对评论文本进行分词、编码、训练，并支持模型评估和交互式预测。

## 项目简介

项目主要流程如下：

1. 读取原始中文商品评论数据；
2. 使用 Jieba 对评论文本进行分词；
3. 构建词表 `vocab.txt`；
4. 将文本编码为固定长度的 token 序列；
5. 使用 LSTM 模型进行二分类训练；
6. 保存最佳模型；
7. 支持测试集评估和命令行预测。

## 项目结构

```text
03-LSTM/
├── data/
│   ├── raw/
│   │   └── online_shopping_10_cats.csv      # 原始数据集
│   └── processed/
│       ├── train.jsonl                      # 处理后的训练集
│       └── test.jsonl                       # 处理后的测试集
├── logs/                                    # TensorBoard 训练日志
├── models/
│   ├── best_model.pth                       # 训练得到的最佳模型
│   └── vocab.txt                            # 词表文件
├── src/
│   ├── config.py                            # 配置文件
│   ├── dataset.py                           # Dataset 和 DataLoader
│   ├── evaluate.py                          # 模型评估
│   ├── model.py                             # LSTM 模型结构
│   ├── predict.py                           # 预测脚本
│   ├── process.py                           # 数据预处理脚本
│   ├── tokenizer.py                         # Jieba 分词与编码
│   └── train.py                             # 模型训练脚本
└── README.md
```

## 环境要求

建议使用 Python 3.10 或以上版本。

主要依赖：

```bash
pip install torch pandas scikit-learn jieba tqdm tensorboard
```

如果你想使用虚拟环境，可以执行：

```bash
python -m venv .venv
```

Windows：

```bash
.venv\\Scripts\\activate
```

macOS / Linux：

```bash
source .venv/bin/activate
```

然后安装依赖：

```bash
pip install torch pandas scikit-learn jieba tqdm tensorboard
```

## 使用方法

### 1. 数据预处理

在项目根目录下执行：

```bash
python src/process.py
```

该命令会完成以下操作：

- 读取 `data/raw/online_shopping_10_cats.csv`；
- 按比例抽样数据；
- 划分训练集和测试集；
- 构建词表 `models/vocab.txt`；
- 生成 `data/processed/train.jsonl` 和 `data/processed/test.jsonl`。

### 2. 训练模型

```bash
python src/train.py
```

训练过程中会：

- 自动检测是否可以使用 GPU；
- 使用 `BCEWithLogitsLoss` 作为损失函数；
- 使用 Adam 优化器；
- 将训练日志保存到 `logs/`；
- 将最佳模型保存为 `models/best_model.pth`。

### 3. 查看 TensorBoard 训练日志

```bash
tensorboard --logdir logs
```

启动后，在浏览器中打开终端提示的地址即可查看 loss 曲线。

### 4. 评估模型

```bash
python src/evaluate.py
```

该脚本会加载 `models/best_model.pth` 和测试集数据，并输出模型准确率。

### 5. 交互式预测

```bash
python src/predict.py
```

运行后输入一条中文评论，模型会输出正向或负向评价，例如：

```text
欢迎使用情感分析模型(输入quit退出)
> 这个商品质量很好，下次还会购买
0.91:正向评价
> 太差了，物流慢，质量也不好
0.08:负向评价
```

输入 `quit` 退出程序。

## 模型说明

模型结构位于 `src/model.py`，核心结构如下：

- `Embedding`：将词索引转换为词向量；
- `LSTM`：提取评论文本的序列特征；
- `Linear`：输出二分类结果；
- `Sigmoid`：在预测阶段将输出转换为正向概率。

主要配置位于 `src/config.py`：

```python
seq_len = 128
batch_size = 64
embedding_dim = 128
hidden_size = 256
learning_rate = 1e-3
epochs = 20
```

## 数据说明

数据集文件位于：

```text
data/raw/online_shopping_10_cats.csv
```

项目使用其中的：

- `review`：评论文本；
- `label`：情感标签。

其中 `label` 用于表示评论情感类别，模型最终完成二分类预测。

## GitHub 上传建议

建议上传核心代码和说明文件，训练日志、缓存文件可以不上传。

推荐保留：

```text
src/
data/raw/online_shopping_10_cats.csv
data/processed/train.jsonl
data/processed/test.jsonl
models/vocab.txt
models/best_model.pth
README.md
```

建议忽略：

```text
__pycache__/
logs/
*.pyc
```

如果模型文件或数据集较大，也可以不上传 `models/best_model.pth` 和 `data/`，改为在 README 中说明如何重新生成。

## 后续优化方向

- 增加验证集，避免只根据训练 loss 保存模型；
- 在评估阶段统计完整测试集准确率、召回率、F1 值；
- 支持双向 LSTM；
- 使用预训练词向量或 Transformer 模型提升效果；
- 增加 `requirements.txt` 方便安装依赖。

## License

本项目仅用于学习和课程实验，可根据需要自行补充开源许可证。
