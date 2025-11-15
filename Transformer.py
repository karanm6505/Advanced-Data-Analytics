"""Transformer-based baseline for multivariate time-series forecasting.

This script trains a sequence-to-one Transformer model on the Beijing PM2.5
pollution dataset (multivariate) and reports RMSE/MAE metrics. It is designed
as the Transformer component of the Phase 1 correlational baselines.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUMERIC_COLUMNS = ["pollution", "dew", "temp", "press", "wnd_spd", "snow", "rain"]
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "LSTM-Multivariate_pollution.csv"
DEFAULT_TEST_PATH = Path(__file__).resolve().parent / "pollution_test_data1.csv"
TORCH_SEED = 42


@dataclass
class DatasetConfig:
    csv_path: Path
    target_col: str = "pollution"
    input_window: int = 48
    forecast_horizon: int = 1
    test_ratio: float = 0.2
    test_csv_path: Optional[Path] = None



def seed_everything(seed: int = TORCH_SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _sanitize_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    subset = df[columns].copy()
    for col in columns:
        series = subset[col].astype(str).str.strip()
        series = series.replace({"": np.nan, "nan": np.nan})
        series = series.str.replace(r"[^0-9.+-]", "", regex=True)
        series = series.replace({"": np.nan})
        subset[col] = pd.to_numeric(series, errors="coerce")
    return subset.dropna()


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values("date").set_index("date")
    return _sanitize_numeric_frame(df, NUMERIC_COLUMNS)


def load_test_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    available_cols = set(df.columns)
    missing = set(NUMERIC_COLUMNS) - available_cols
    if missing:
        raise ValueError(f"Test file is missing columns: {sorted(missing)}")
    return _sanitize_numeric_frame(df, NUMERIC_COLUMNS)


def create_sequences(
    array: np.ndarray,
    input_window: int,
    horizon: int,
    target_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    features, targets = [], []
    max_start = len(array) - input_window - horizon + 1
    for start in range(max_start):
        end = start + input_window
        future_index = end + horizon - 1
        features.append(array[start:end])
        targets.append(array[future_index, target_idx])
    return np.stack(features), np.array(targets)


def prepare_dataloaders(
    config: DatasetConfig,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, StandardScaler, int]:
    df = load_dataframe(config.csv_path)
    values = df.values
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values)

    target_idx = NUMERIC_COLUMNS.index(config.target_col)

    if config.test_csv_path is None:
        split_index = int(len(scaled_values) * (1 - config.test_ratio))
        train_array = scaled_values[:split_index]
        test_array = scaled_values[split_index - config.input_window :]
    else:
        train_array = scaled_values
        test_df = load_test_dataframe(config.test_csv_path)
        test_values = scaler.transform(test_df.values)
        test_array = test_values

    train_x, train_y = create_sequences(train_array, config.input_window, config.forecast_horizon, target_idx)
    test_x, test_y = create_sequences(test_array, config.input_window, config.forecast_horizon, target_idx)

    train_dataset = SequenceDataset(train_x, train_y)
    test_dataset = SequenceDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader, scaler, target_idx


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.decoder = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(src)
        x = self.positional_encoding(x)
        encoded = self.encoder(x)
        summary_vector = encoded[:, -1, :]
        output = self.decoder(summary_vector)
        return output.squeeze(-1)


# ---------------------------------------------------------------------------
# Training & evaluation routines
# ---------------------------------------------------------------------------


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    running_loss = 0.0
    for features, targets in loader:
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    return running_loss / len(loader.dataset)


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler,
    target_idx: int,
) -> Tuple[float, float]:
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for features, targets in loader:
            outputs = model(features.to(DEVICE))
            preds.append(outputs.cpu().numpy())
            truths.append(targets.numpy())
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)

    target_mean = scaler.mean_[target_idx]
    target_scale = scaler.scale_[target_idx]
    preds_unscaled = preds * target_scale + target_mean
    truths_unscaled = truths * target_scale + target_mean

    rmse = math.sqrt(mean_squared_error(truths_unscaled, preds_unscaled))
    mae = mean_absolute_error(truths_unscaled, preds_unscaled)
    return rmse, mae


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer baseline on the multivariate pollution dataset.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to the multivariate CSV file.")
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH, help="Optional path to an external test CSV file.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for training.")
    parser.add_argument("--input-window", type=int, default=48, help="Length of the encoder window in time steps.")
    parser.add_argument("--forecast-horizon", type=int, default=1, help="Number of steps ahead to forecast.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for the optimizer.")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer model dimension.")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--layers", type=int, default=2, help="Transformer encoder layers.")
    parser.add_argument("--ff-dim", type=int, default=128, help="Feedforward dimension inside Transformer blocks.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything()

    config = DatasetConfig(
        csv_path=args.data_path,
        target_col="pollution",
        input_window=args.input_window,
        forecast_horizon=args.forecast_horizon,
        test_csv_path=args.test_path,
    )

    train_loader, test_loader, scaler, target_idx = prepare_dataloaders(config, batch_size=args.batch_size)

    model = TimeSeriesTransformer(
        num_features=len(NUMERIC_COLUMNS),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        if epoch % 5 == 0 or epoch == args.epochs:
            rmse, mae = evaluate_metrics(model, test_loader, scaler, target_idx)
            print(
                f"Epoch {epoch:03d}: train_loss={train_loss:.4f} | test_RMSE={rmse:.3f} | test_MAE={mae:.3f}",
                flush=True,
            )

    train_rmse, train_mae = evaluate_metrics(model, train_loader, scaler, target_idx)
    test_rmse, test_mae = evaluate_metrics(model, test_loader, scaler, target_idx)

    print("\nFinal evaluation:")
    print(f"Train RMSE: {train_rmse:.3f}\tTrain MAE: {train_mae:.3f}")
    print(f"Test RMSE:  {test_rmse:.3f}\tTest MAE:  {test_mae:.3f}")


if __name__ == "__main__":
    main()
