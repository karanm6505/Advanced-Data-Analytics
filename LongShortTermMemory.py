import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Fixing a random seed
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

# Read CSV - use relative path
csv_path = "daily-minimum-temperatures-in-me.csv"
df = pd.read_csv(csv_path)

# Normalize column name to 'value' (the dataset header is "Daily minimum temperatures")
if 'Daily minimum temperatures' in df.columns:
    df.rename(columns={'Daily minimum temperatures': 'value'}, inplace=True)
elif 'value' not in df.columns:
    # Fallback: assume the file has two columns [Date, VALUE]
    df.columns = ['Date', 'value']

# Coerce non-numeric entries (e.g. rows with leading '?') to NaN and drop them
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df.dropna(subset=['value'], inplace=True)
df.reset_index(drop=True, inplace=True)

value_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = value_scaler.fit_transform(df[['value']]).flatten()

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
X, Y = create_sequences(scaled_values, seq_len)

# -------------------------
# Train-test split (no shuffle for time-series)
# -------------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
Y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(-1)

y_train = Y_train.squeeze().numpy()
y_test = Y_test.squeeze().numpy()

# -------------------------
# Define LSTM model
# -------------------------
class LongShortTermMemory(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LongShortTermMemory, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# -------------------------
# Train model
# -------------------------
model = LongShortTermMemory(input_size=1, hidden_size=32, num_layers=1, output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.HuberLoss(delta=1.0)

num_epochs = 500
train_losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, Y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.6f}')

# -------------------------
# Evaluate on Test Data
# -------------------------
model.eval()
with torch.no_grad():
    predicted_train_scaled = model(X_train)
    predicted_test_scaled = model(X_test)
    test_loss = criterion(predicted_test_scaled, Y_test).item()
    predicted_train = predicted_train_scaled.squeeze(-1).cpu().numpy()
    predicted_test = predicted_test_scaled.squeeze(-1).cpu().numpy()

    train_actual_scaled = Y_train.squeeze(-1).cpu().numpy()
    test_actual_scaled = Y_test.squeeze(-1).cpu().numpy()

    predicted_train = value_scaler.inverse_transform(predicted_train.reshape(-1, 1)).squeeze()
    predicted_test = value_scaler.inverse_transform(predicted_test.reshape(-1, 1)).squeeze()
    train_actual = value_scaler.inverse_transform(train_actual_scaled.reshape(-1, 1)).squeeze()
    test_actual = value_scaler.inverse_transform(test_actual_scaled.reshape(-1, 1)).squeeze()

print(f"\nTest Loss: {test_loss:.6f}")

# -------------------------
# Plot Actual vs Predicted
# -------------------------
train_length = len(train_actual)
test_length = len(test_actual)
t_lstm = np.arange(train_length + test_length)

plt.figure(figsize=(12, 5))
plt.plot(t_lstm[:train_length], train_actual, label='Train Actual (LSTM)', color='blue')
plt.plot(t_lstm[train_length:], test_actual, label='Test Actual (LSTM)', color='green')
plt.plot(t_lstm[:len(predicted_train)], predicted_train, label='Train Predicted (LSTM)', color='red', linestyle='--')
plt.plot(t_lstm[train_length:train_length+len(predicted_test)], predicted_test, label='Test Predicted (LSTM)', color='orange', linestyle='--')

# Optional shading to highlight test region
plt.axvspan(train_length, train_length+test_length, color='gray', alpha=0.1)

plt.title('LSTM: Train + Test Actual vs Predicted')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()
