import torch
from torch import nn
import config

class ReviewAnalyzeModel(nn.Module):
    def __init__(self,vocab_size,padding_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,config.embedding_dim,padding_idx=padding_index)
        self.lstm = nn.LSTM(input_size=config.embedding_dim,hidden_size=config.hidden_size,batch_first=True)
        self.linear = nn.Linear(config.hidden_size,1)

    def forward(self,x): # x.size:[batch_size,seq_len]
        embed = self.embedding(x) # embed.size:[batch_size,seq_len,embedding_dim]
        output,(_,_) = self.lstm(embed) # output.size:[batch_size,seq_len,hidden_size]
        batch_index = torch.arange(0,output.shape[0])
        lengths = (x!=self.embedding.padding_idx).sum(dim=1) # 0 0 0 <pad> --> 1 1 1 0 --> 需要拿到最后一个非<pad>的隐藏层内容
        last_hidden = output[batch_index,lengths-1]

        output = self.linear(last_hidden).squeeze(-1) # output.shape:[batch_size]
        return output