import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import JiebaTokenizer
import config

def process():
    print("开始处理数据")
    df = pd.read_csv(config.raw_data_path/'online_shopping_10_cats.csv',usecols=['label','review'],
                     encoding='utf-8').dropna().sample(frac=0.1)
    # 划分数据集  stratify实现分层抽样
    train_df,test_df = train_test_split(df,test_size=0.2,stratify=df['label'])

    # 构建词表
    JiebaTokenizer.build_vocab(train_df['review'].tolist(),config.models_path/'vocab.txt')

    # 构建tokenizer
    tokenizer = JiebaTokenizer.from_vocab(config.models_path/'vocab.txt')

    # 计算序列长度
    #print(train_df['review'].apply(lambda x: len(tokenizer.tokenize(x))).quantile(0.95))

    # 构建训练集
    train_df['review'] = train_df['review'].apply(lambda x: tokenizer.encode(x,config.seq_len))

    # 保存数据集
    train_df.to_json(config.processed_data_path/'train.jsonl',orient='records',lines=True)

    # 构建测试集
    test_df['review'] = test_df['review'].apply(lambda x: tokenizer.encode(x,config.seq_len))

    # 保存测试集
    test_df.to_json(config.processed_data_path/'test.jsonl',orient='records',lines=True)
    print("数据处理结束")

if __name__ == '__main__':
    process()