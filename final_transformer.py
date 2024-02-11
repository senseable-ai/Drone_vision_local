import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from modified_data_split import load_preprocessed_data
import wandb
import numpy as np

# Initialize wandb
wandb.init(project="lane_change_prediction_optuna", entity="imij0522")

# Load data
X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()

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
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, src):
        src = src.permute(1, 0, 2)  # Convert (batch_size, seq_len, feature) to (seq_len, batch_size, feature)
        src = self.input_proj(src)  # Linear projection to model dimension
        src = self.pos_encoder(src)
        output = self.transformer(src, src)
        output = self.output_proj(output[-1, :, :])  # Project output to target size
        return output

# Set hyperparameters
config = {
    "input_dim": 34,
    "d_model": 128,
    "nhead": 4,
    "num_encoder_layers": 2,
    "num_decoder_layers": 3,
    "dim_feedforward": 2048,
    "dropout": 0.0001958233602860071,
    "batch_size": 128,
    "learning_rate": 0.0021974989507080453,
    "epochs": 10
}

# Prepare data loaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float))
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

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

# Load the provided model state dictionary
model_path = '/home/user/drone-vision/drone_vision_local/lane_change_model_optuna.pth'
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)

# Initialize loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# Training and validation loop
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

    wandb.log({"train_loss": train_loss / len(train_loader)})

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()

    wandb.log({"val_loss": val_loss / len(val_loader)})

    print(f'Epoch {epoch+1}: Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}')

# Test loop
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float))
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

test_predictions = []
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs[0].to(device)
        outputs = model(inputs)
        test_predictions.extend(outputs.squeeze().cpu().numpy())

# Log test predictions to wandb
wandb.log({"test_predictions": test_predictions})

# Save the test predictions to a file
np.save('test_predictions.npy', np.array(test_predictions))

# Finish the wandb run
wandb.finish()
