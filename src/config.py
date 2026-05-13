from pathlib import Path

# print(__file__) # 识别该文件的绝对目录
root_path = Path(__file__).parent.parent

raw_data_path = root_path / "data" / "raw"
processed_data_path = root_path / "data" / "processed"
logs_path = root_path / "logs"
models_path = root_path / "models"

seq_len = 128
batch_size = 64
embedding_dim = 128
hidden_size = 256
learning_rate = 1e-3
epochs = 20