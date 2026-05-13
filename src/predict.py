import jieba
from tokenizer import JiebaTokenizer
import config
import torch
from model import ReviewAnalyzeModel

def predict_batch(model,inputs):
    """
    批量预测
    :param model:模型
    :param inputs:输入 shape[batch_size,seq_len]
    :return:预测结果 shape[batch_size]
    """
    model.eval()
    with torch.no_grad():
        model_output = model(inputs)  # shape:[batch_size,vocab_size] ->[1,vocab_size]
    batch_result = torch.sigmoid(model_output)
    return batch_result.tolist()

def predict(text,model,tokenizer,device):
    # 处理输入
    indexs = tokenizer.encode(text,seq_len = config.seq_len)
    input_tensor = torch.tensor([indexs],dtype=torch.long)
    input_tensor = input_tensor.to(device)
    # 预测逻辑
    batch_result = predict_batch(model,input_tensor)
    return batch_result[0]

def run_predict():

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 词表
    tokenizer = JiebaTokenizer.from_vocab(config.models_path / 'vocab.txt')
    # 模型
    model = ReviewAnalyzeModel(tokenizer.vocab_size,tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.models_path / 'best_model.pth'))

    print("欢迎使用情感分析模型(输入quit退出)")
    while True:
        user_input = input("> ")
        if user_input == "quit":
            print("即将离开系统")
            break
        if user_input.strip() == "":
            print("请输入内容")
            continue
        result = predict(user_input,model,tokenizer,device)
        if result > 0.5:
            print(f"{result}:正向评价")
        else:
            print(f"{result}:负向评价")

if __name__ == '__main__':
    run_predict()