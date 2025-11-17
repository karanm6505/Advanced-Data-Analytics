"""Linear model with causal regularization penalty."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import math
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
DEFAULT_CAUSAL_ADJ_PATH = Path(__file__).resolve().parent / "causal_graph_lagged_adjacency.csv"
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


def _sanitize_numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    subset = df[list(columns)].copy()
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

    def __len__(self) -> int:  # pragma: no cover - simple proxy
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


def create_sequences(array: np.ndarray, input_window: int, horizon: int, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    features, targets = [], []
    max_start = len(array) - input_window - horizon + 1
    for start in range(max_start):
        end = start + input_window
        future_index = end + horizon - 1
        features.append(array[start:end])
        targets.append(array[future_index, target_idx])
    return np.stack(features), np.array(targets)


def prepare_dataloaders(config: DatasetConfig, batch_size: int) -> Tuple[DataLoader, DataLoader, StandardScaler, int]:
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


def parse_variable_name(name: str) -> Optional[Tuple[str, int]]:
    if "_t" not in name:
        return None
    feature, _, lag_part = name.partition("_t")
    lag = 0
    if lag_part.startswith("-"):
        try:
            lag = int(lag_part[1:])
        except ValueError:
            return None
    return feature, lag


def build_allowed_mask(input_window: int, adj_path: Path) -> torch.Tensor:
    if not adj_path.exists():
        return torch.ones(input_window * len(NUMERIC_COLUMNS), dtype=torch.bool)
    adjacency = pd.read_csv(adj_path, index_col=0)
    if "pollution_t" not in adjacency.columns:
        return torch.ones(input_window * len(NUMERIC_COLUMNS), dtype=torch.bool)
    allowed_tokens = set()
    parents = adjacency.index[adjacency["pollution_t"] != 0]
    for parent in parents:
        parsed = parse_variable_name(parent)
        if not parsed:
            continue
        feature, lag = parsed
        if feature in NUMERIC_COLUMNS:
            allowed_tokens.add((feature, lag))
    mask = torch.zeros(input_window * len(NUMERIC_COLUMNS), dtype=torch.bool)
    for time_idx in range(input_window):
        lag = input_window - 1 - time_idx
        for feat_idx, feature in enumerate(NUMERIC_COLUMNS):
            token_idx = time_idx * len(NUMERIC_COLUMNS) + feat_idx
            if (feature, lag) in allowed_tokens:
                mask[token_idx] = True
    return mask


class LinearCausalRegressor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def evaluate_metrics(model: nn.Module, loader: DataLoader, scaler: StandardScaler, target_idx: int) -> Tuple[float, float]:
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for features, targets in loader:
            flat = features.view(features.size(0), -1).to(DEVICE)
            outputs = model(flat).cpu().numpy()
            preds.append(outputs)
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


def train_model(
    model: LinearCausalRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    penalty_mask: torch.Tensor,
    penalty_lambda: float,
    epochs: int,
) -> None:
    criterion = nn.MSELoss()
    penalty_mask = penalty_mask.to(DEVICE)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for features, targets in loader:
            flat = features.view(features.size(0), -1).to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()
            preds = model(flat)
            mse = criterion(preds, targets)
            weights = model.linear.weight.squeeze(0)
            penalty = penalty_lambda * torch.abs(weights[~penalty_mask]).sum()
            loss = mse + penalty
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
        avg_loss = running_loss / len(loader.dataset)
        print(f"Epoch {epoch:03d}: train_loss={avg_loss:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear causal regularization baseline.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--causal-adj-path", type=Path, default=DEFAULT_CAUSAL_ADJ_PATH)
    parser.add_argument("--lambda-penalty", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--input-window", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything()

    config = DatasetConfig(
        csv_path=args.data_path,
        target_col="pollution",
        input_window=args.input_window,
        forecast_horizon=1,
        test_csv_path=args.test_path,
    )
    train_loader, test_loader, scaler, target_idx = prepare_dataloaders(config, batch_size=args.batch_size)

    penalty_mask = build_allowed_mask(args.input_window, args.causal_adj_path)
    input_dim = args.input_window * len(NUMERIC_COLUMNS)
    model = LinearCausalRegressor(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_model(model, train_loader, optimizer, penalty_mask, args.lambda_penalty, args.epochs)

    train_rmse, train_mae = evaluate_metrics(model, train_loader, scaler, target_idx)
    test_rmse, test_mae = evaluate_metrics(model, test_loader, scaler, target_idx)

    print("\nFinal evaluation:")
    print(f"Train RMSE: {train_rmse:.3f}\tTrain MAE: {train_mae:.3f}")
    print(f"Test RMSE:  {test_rmse:.3f}\tTest MAE:  {test_mae:.3f}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
