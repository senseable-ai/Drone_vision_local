import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from feature.test_data_split import load_preprocessed_data
import random
import wandb
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, confusion_matrix
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
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
# Initialize wandb
wandb.init(project="test2", entity="finnn")
# Load data (You should define load_preprocessed_data function or replace it with actual data loading)
X_train, X_test, y_train, y_test = load_preprocessed_data()
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.output_proj = nn.Linear(d_model, 1)  # Output layer for binary classification
    def forward(self, src):
        src = src.unsqueeze(1)
        src = src.permute(1, 0, 2)
        src = self.input_proj(src)
        src = self.pos_encoder(src)
        output = self.transformer(src, src)
        output = self.output_proj(output[-1, :, :])
        return output
# Set hyperparameters
config = {
    "input_dim": 31,
    "d_model": 128,
    "nhead": 4,
    "num_encoder_layers": 2,
    "num_decoder_layers": 3,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 10
}
def pad_and_create_tensors(sequences, labels, max_len=None):
    # 시퀀스를 고정 길이로 패딩하고 텐서로 변환하는 함수
    padded_sequences = []
    for seq in sequences:
        # numpy 배열을 텐서로 변환
        tensor = torch.tensor(seq, dtype=torch.float)
        padded_sequences.append(tensor)
    # 모든 시퀀스를 최대 길이로 패딩
    padded_sequences = pad_sequence(padded_sequences, batch_first=True, padding_value=0)
    # 레이블 텐서 생성
    labels_tensor = torch.tensor(labels, dtype=torch.float)
    return padded_sequences, labels_tensor
# 데이터 로드 및 패딩
X_train_padded, y_train_tensor = pad_and_create_tensors(X_train, y_train)
X_test_padded, y_test_tensor = pad_and_create_tensors(X_test, y_test)
# 데이터 로더 생성
train_dataset = TensorDataset(X_train_padded, y_train_tensor)
test_dataset = TensorDataset(X_test_padded, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
# Initialize model and move it to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformer(
    config["input_dim"],
    config["d_model"],
    config["nhead"],
    config["num_encoder_layers"],
    config["num_decoder_layers"],
    config["dim_feedforward"],
    config["dropout"]
).to(device)
# Initialize loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
# Training loop
for epoch in range(config["epochs"]):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f'Epoch {epoch+1}: Train Loss: {train_loss / len(train_loader)}')
    # Validation loop can be added here
# Test loop and metrics calculation
test_dataset = TensorDataset(X_test_padded, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
true_labels = []
pred_scores = []  # 변경: 예측 확률을 저장하기 위해 pred_labels에서 pred_scores로 이름 변경
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predictions = torch.sigmoid(outputs).squeeze()
        true_labels.extend(labels.cpu().numpy())
        pred_scores.extend(predictions.cpu().numpy())  # 변경: 이진 예측 대신 확률 저장
# Calculate metrics
accuracy = accuracy_score(true_labels, (np.array(pred_scores) > 0.5).astype(int))
precision = precision_score(true_labels, (np.array(pred_scores) > 0.5).astype(int))
recall = recall_score(true_labels, (np.array(pred_scores) > 0.5).astype(int))
f1 = f1_score(true_labels, (np.array(pred_scores) > 0.5).astype(int))
cm = confusion_matrix(true_labels, (np.array(pred_scores) > 0.5).astype(int))
TP, FP, FN, TN = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
# Log metrics to W&B
wandb.log({
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "TP": TP,
    "FN": FN,
    "FP": FP,
    "TN": TN,
    "Epoch Loss": train_loss / len(train_loader)  # 추가: 에폭 손실 기록
})
# ROC Curve 계산 및 그리기
fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
roc_auc = roc_auc_score(true_labels, pred_scores)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
# Log ROC Curve to W&B
wandb.log({"ROC Curve": wandb.Image(plt)})
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
print(f"Confusion Matrix: TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
# Finish the wandb run
wandb.finish()