import torch
import config
from model import ReviewAnalyzeModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import JiebaTokenizer

def evaluate(model, test_dataloader, device):
    total_count = 0
    correct_count = 0
    for inputs,targets in test_dataloader:
        inputs = inputs.to(device) # shape:[batch_size,seq_len]
        targets = targets.tolist() # shape:[batch_size]
        # outputs = model(inputs) # [batch_size,vocab_size]

        batch_result = predict_batch(model, inputs)
        for result,target in zip(batch_result,targets):
            result = 1 if result > 0.5 else 0
            if result == target:
                correct_count += 1
            total_count += 1
        return correct_count / total_count

def run_evaluate():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 词表
    tokenizer = JiebaTokenizer.from_vocab(config.models_path / "vocab.txt")
    print("词表加载成功")
    # 模型
    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size,padding_index=tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.models_path / 'best_model.pth'))
    print("模型加载成功")
    # 数据集
    test_dataloader = get_dataloader(train=False)
    # 评估逻辑
    acc = evaluate(model, test_dataloader, device)
    print("评估结果")
    print(f"acc:{acc}")

if __name__ == '__main__':
    run_evaluate()