import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from feature.modified_data_split import load_preprocessed_data
import wandb
import optuna
import numpy as np
import random

# 시드 고정 함수
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 시드 고정
seed_everything()

# GPU 사용 가능 여부 확인 및 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# wandb 초기화
wandb.init(project="optuna", entity="imij0522")

# 데이터 로드
X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers, 
                                          dim_feedforward=dim_feedforward, 
                                          dropout=dropout)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, src):
        src = src.permute(1, 0, 2)
        src = self.input_proj(src)
        output = self.transformer(src, src)
        output = self.output_proj(output[-1, :, :])
        return output

# Objective function moved outside the class
def objective(trial):
    seed_everything()

    # 하이퍼파라미터 제안
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 4)
    num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 4)
    d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [512, 1024, 2048])
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    model = TimeSeriesTransformer(input_dim=34, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 데이터 로더 설정
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 학습 루프
    for epoch in range(10):  # 하이퍼파라미터 탐색 시 에폭 수를 줄입니다.
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
    # 검증 루프
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

best_params = study.best_trial.params

model = TimeSeriesTransformer(input_dim=34, d_model=best_params['d_model'], nhead=best_params['nhead'], num_encoder_layers=best_params['num_encoder_layers'], num_decoder_layers=best_params['num_decoder_layers'], dim_feedforward=best_params['dim_feedforward'], dropout=best_params['dropout']).to(device)

# 최적 하이퍼파라미터로 모델 학습 및 검증 루프 반복

model_path = 'lane_change_model_optuna.pth'
torch.save(model.state_dict(), model_path)
