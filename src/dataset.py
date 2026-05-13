import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config
# 定义dataset
class ReviewAnalyzeDataset(Dataset):
    def __init__(self,path):
        self.data = pd.read_json(path,orient="records",lines=True).to_dict(orient="records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        input_tensor = torch.tensor(self.data[index]['review'],dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['label'],dtype=torch.float)
        return input_tensor,target_tensor

# 提供一个获取dataloader的方法
def get_dataloader(train=True):
    path = config.processed_data_path / ('train.jsonl' if train else 'test.jsonl')
    dataset = ReviewAnalyzeDataset(path)
    return DataLoader(dataset,batch_size = config.batch_size,shuffle=True) # shuffle在每个epoch开始时随机打乱数据集的样本顺序

if __name__ == '__main__':
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)
    print(len(train_dataloader)) # 106165
    print(len(test_dataloader)) #  26540

    for input_tensor,target_tensor in train_dataloader:
        print(input_tensor.shape) # [batch_size,seq_len] torch.Size([32, 5])
        print(target_tensor.shape) # [batch_size]   torch.Size([32])
        break