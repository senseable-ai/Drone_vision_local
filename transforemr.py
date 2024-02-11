import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# 수정된 data_split.py에서 전처리된 데이터 로드 함수 불러오기
from data_split import load_preprocessed_data

# 데이터 로드
X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()

# 모델 정의
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers, 
                                          dim_feedforward=dim_feedforward, 
                                          dropout=dropout)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, 1) # 예제는 1차원 출력을 가정

    def forward(self, src):
        src = self.input_proj(src)
        output = self.transformer(src, src)
        output = self.output_proj(output[:, -1, :])
        return output

# 하이퍼파라미터 설정
input_dim = 34
d_model = 512
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 2048
dropout = 0.1
batch_size = 64
learning_rate = 0.001
epochs = 10

# 데이터 로더 설정
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 모델, 손실 함수, 옵티마이저 초기화
model = TimeSeriesTransformer(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 및 검증 루프
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 10 == 0:  # Adjust the frequency of printing based on your preference
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Train Loss: {loss.item()}')

    # 검증
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()
    print(f'Epoch {epoch+1}, Average Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}')
