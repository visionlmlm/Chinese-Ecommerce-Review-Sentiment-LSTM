import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloader
from tokenizer import JiebaTokenizer
from model import ReviewAnalyzeModel

def train_one_epoch(model,dataloader,loss_fn,optimizer,device):
    total_loss = 0
    model.train()
    for inputs,targets in tqdm(dataloader,desc="Training"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs,targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def train():

    # 设备
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据
    dataloader = get_dataloader()
    # 分词器
    tokenizer = JiebaTokenizer.from_vocab(config.models_path/'vocab.txt')
    # 模型
    model =ReviewAnalyzeModel(tokenizer.vocab_size,tokenizer.pad_token_index).to(device)
    # 损失函数
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(),lr = config.learning_rate)
    # tensorBoard writer  tensorboard --logdir F:\NLP_test\03-LSTM\logs
    writer = SummaryWriter(log_dir=config.logs_path/time.strftime('%Y%m%d_%H%M%S'))

    best_loss = float('inf')
    for epoch in range(1,config.epochs+1):
        print(f"=============Epoch {epoch}=============")
        loss = train_one_epoch(model,dataloader,loss_fn,optimizer,device)
        print(f"loss: {loss:.4f}")
        writer.add_scalar('loss',loss,epoch)
        # 保存模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(),config.models_path/'best_model.pth')
            print("Saving best model")
    writer.close()

if __name__ == '__main__':
    train()