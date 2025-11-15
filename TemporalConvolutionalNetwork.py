import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Fix random seed
np.random.seed(67)
torch.manual_seed(67)

# -------------------------
# Generate synthetic time series data
# -------------------------
t = np.arange(0, 200, 0.1)
y = np.zeros_like(t)
for i in range(0, len(t), 200):
    freq = np.random.uniform(0.05, 0.2)
    amp = np.random.uniform(0.5, 1.5)
    noise = np.random.randn(min(200, len(t)-i)) * np.random.uniform(0.1, 0.4)
    y[i:i+200] = amp * np.sin(freq * t[i:i+200]) + noise

# Read CSV - use raw string for Windows path to avoid unicode-escape issues
# Change csv_path to the file you want to use. The parser below will take the
# first two columns as [date, value] regardless of their header names.
csv_path = r"C:\Users\suneela gabbita\Downloads\Advanced-Data-Analytics-main\Advanced-Data-Analytics-main\Electric_Production.csv"
df = pd.read_csv(csv_path, header=0)

# Defensive column handling: take the first two columns as date and value
if df.shape[1] < 2:
    raise ValueError(f"CSV at {csv_path} must have at least two columns (date, value)")

orig_cols = [c.strip() for c in df.columns]
date_col, val_col = orig_cols[0], orig_cols[1]

# Parse date column (coerce errors to NaT)
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

# Clean value column: remove non-numeric characters except minus and dot, then coerce
df[val_col] = df[val_col].astype(str).str.strip()
df[val_col] = df[val_col].str.replace(r"[^0-9eE+\-\.]", "", regex=True)
df[val_col] = pd.to_numeric(df[val_col], errors='coerce')

# Rename to standardized names and drop invalid rows
df.rename(columns={date_col: 'Date', val_col: 'value'}, inplace=True)
df.dropna(subset=['Date', 'value'], inplace=True)
df.reset_index(drop=True, inplace=True)


# -------------------------
# Create sequences
# -------------------------
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs), np.array(ys)

seq_len = 20
X, Y = create_sequences(df['value'].values, seq_len)

# Train-test split (no shuffle for time-series)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # (batch, 1, seq_len)
Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
Y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(-1)

# -------------------------
# Define TCN model
# -------------------------
class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, num_layers):
        super(TemporalConvolutionalNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            conv = nn.Conv1d(input_channels, output_channels, kernel_size,
                             padding=padding, dilation=dilation)
            self.layers.append(conv)
            input_channels = output_channels
        self.fc = nn.Linear(output_channels, 1)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = x[:, :, -1]  # last time step
        x = self.fc(x)
        return x

# -------------------------
# Initialize and Train Model
# -------------------------
model = TemporalConvolutionalNetwork(input_channels=1, output_channels=32, kernel_size=3, num_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.HuberLoss()

num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, Y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.6f}")

# -------------------------
# Evaluate on Test Data
# -------------------------
model.eval()
with torch.no_grad():
    train_preds = model(X_train)
    test_preds = model(X_test)
    test_loss = criterion(test_preds, Y_test).item()

print(f"\nTest Loss: {test_loss:.6f}")

# -------------------------
# Plot Actual vs Predicted
# -------------------------
t_all = np.arange(len(Y_train) + len(Y_test))

plt.figure(figsize=(12, 5))
plt.plot(t_all[:len(Y_train)], Y_train.numpy(), label='Train Actual (TCN)', color='blue')
plt.plot(t_all[len(Y_train):], Y_test.numpy(), label='Test Actual (TCN)', color='green')
plt.plot(t_all[:len(Y_train)], train_preds.numpy(), label='Train Predicted (TCN)', color='red', linestyle='--')
plt.plot(t_all[len(Y_train):len(Y_train)+len(Y_test)], test_preds.numpy(), label='Test Predicted (TCN)', color='orange', linestyle='--')

plt.axvspan(len(Y_train), len(Y_train)+len(Y_test), color='gray', alpha=0.1)
plt.title('TCN: Train + Test Actual vs Predicted')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()
