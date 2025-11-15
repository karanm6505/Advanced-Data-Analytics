"""LSTM baseline for the multivariate pollution forecasting task."""

from __future__ import annotations

import argparse
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
    test_csv_path: Optional[Path] = DEFAULT_TEST_PATH


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


class LongShortTermMemory(nn.Module):
    def __init__(self, num_features: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        summary = output[:, -1, :]
        return self.head(summary).squeeze(-1)


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    running_loss = 0.0
    for features, targets in loader:
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, targets)
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

    rmse = float(np.sqrt(mean_squared_error(truths_unscaled, preds_unscaled)))
    mae = float(mean_absolute_error(truths_unscaled, preds_unscaled))
    return rmse, mae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM baseline on the multivariate pollution dataset.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to the multivariate CSV file.")
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH, help="Optional path to an external test CSV file.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for training.")
    parser.add_argument("--input-window", type=int, default=48, help="Length of the encoder window in time steps.")
    parser.add_argument("--forecast-horizon", type=int, default=1, help="Number of steps ahead to forecast.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for the optimizer.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden state dimension of the LSTM.")
    parser.add_argument("--layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout applied between LSTM layers.")
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

    model = LongShortTermMemory(
        num_features=len(NUMERIC_COLUMNS),
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        if epoch % 5 == 0 or epoch == args.epochs:
            rmse, mae = evaluate_metrics(model, test_loader, scaler, target_idx)
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} | test_RMSE={rmse:.3f} | test_MAE={mae:.3f}")

    train_rmse, train_mae = evaluate_metrics(model, train_loader, scaler, target_idx)
    test_rmse, test_mae = evaluate_metrics(model, test_loader, scaler, target_idx)

    print("\nFinal evaluation:")
    print(f"Train RMSE: {train_rmse:.3f}\tTrain MAE: {train_mae:.3f}")
    print(f"Test RMSE:  {test_rmse:.3f}\tTest MAE:  {test_mae:.3f}")


if __name__ == "__main__":
    main()
