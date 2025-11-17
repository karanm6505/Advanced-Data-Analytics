"""Transformer-based baseline for multivariate time-series forecasting.

This script trains a sequence-to-one Transformer model on the Beijing PM2.5
pollution dataset (multivariate) and reports RMSE/MAE metrics. It is designed
as the Transformer component of the Phase 1 correlational baselines.
"""
from __future__ import annotations

import argparse
import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


VAR_PATTERN = re.compile(r"^(?P<name>[a-z_]+)_t(?:-(?P<lag>\d+))?$")


def parse_variable_name(name: str) -> Optional[Tuple[str, int]]:
    match = VAR_PATTERN.match(name)
    if not match:
        return None
    feature = match.group("name")
    lag = int(match.group("lag") or 0)
    return feature, lag


def extract_feature_edges(path: Path) -> List[Tuple[str, int, str, int]]:
    if path is None or not path.exists():
        return []
    try:
        adjacency = pd.read_csv(path, index_col=0)
    except Exception:
        return []
    edges: List[Tuple[str, int, str, int]] = []
    for src in adjacency.index:
        for dst in adjacency.columns:
            if adjacency.at[src, dst] == 0:
                continue
            parsed_src = parse_variable_name(src)
            parsed_dst = parse_variable_name(dst)
            if not parsed_src or not parsed_dst:
                continue
            src_feat, src_lag = parsed_src
            dst_feat, dst_lag = parsed_dst
            if src_feat not in NUMERIC_COLUMNS or dst_feat not in NUMERIC_COLUMNS:
                continue
            edges.append((src_feat, src_lag, dst_feat, dst_lag))
    return edges


def infer_max_lag_from_adjacency(path: Path, outcome: str = "pollution_t") -> Optional[int]:
    if path is None or not path.exists():
        return None
    try:
        adjacency = pd.read_csv(path, index_col=0)
    except Exception:
        return None
    if outcome not in adjacency.columns:
        return None
    parent_mask = adjacency[outcome] != 0
    parents = adjacency.index[parent_mask]
    lag_pattern = re.compile(r"t-(\d+)")
    lags = []
    for name in parents:
        match = lag_pattern.search(str(name))
        if match:
            lags.append(int(match.group(1)))
    return max(lags) if lags else None


def build_causal_attention_mask(seq_len: int, max_lag: int) -> torch.Tensor:
    max_lag = max(0, min(max_lag, seq_len - 1))
    mask = torch.full((seq_len, seq_len), float("-inf"))
    for tgt in range(seq_len):
        start = max(0, tgt - max_lag)
        mask[tgt, start : tgt + 1] = 0.0
    return mask


def build_feature_token_attention_mask(
    seq_len: int,
    num_features: int,
    max_lag: Optional[int],
    edges: Sequence[Tuple[str, int, str, int]],
    add_global_token: bool = False,
) -> Tuple[torch.Tensor, List[Dict[str, int | str]]]:
    base_token_count = seq_len * num_features
    token_count = base_token_count + (1 if add_global_token else 0)
    mask = torch.full((token_count, token_count), float("-inf"))
    metadata: List[Dict[str, int | str]] = []
    for time_idx in range(seq_len):
        lag = seq_len - 1 - time_idx
        for feat_idx, feature in enumerate(NUMERIC_COLUMNS):
            metadata.append({"time_idx": time_idx, "lag": lag, "feature": feature, "feat_idx": feat_idx})

    edge_map: Dict[str, List[Tuple[str, int]]] = {}
    for src_feat, src_lag, dst_feat, dst_lag in edges:
        delta = src_lag - dst_lag
        edge_map.setdefault(dst_feat, []).append((src_feat, delta))

    for tgt_idx, tgt_info in enumerate(metadata):
        mask[tgt_idx, tgt_idx] = 0.0
        for src_idx, src_info in enumerate(metadata):
            if src_info["time_idx"] > tgt_info["time_idx"]:
                continue
            if max_lag is not None and (tgt_info["time_idx"] - src_info["time_idx"]) > max_lag:
                continue
            if src_info["feature"] == tgt_info["feature"]:
                mask[tgt_idx, src_idx] = 0.0
                continue
            allowed = False
            for edge_src_feat, lag_delta in edge_map.get(tgt_info["feature"], []):
                if src_info["feature"] != edge_src_feat:
                    continue
                desired_lag = tgt_info["lag"] + lag_delta
                if desired_lag == src_info["lag"]:
                    allowed = True
                    break
            if allowed:
                mask[tgt_idx, src_idx] = 0.0

    if add_global_token:
        global_idx = token_count - 1
        mask[global_idx, :] = 0.0
        mask[global_idx, global_idx] = 0.0
    return mask, metadata


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
        feature_tokenize: bool = False,
        pollution_index: int = 0,
        use_global_token: bool = True,
    ) -> None:
        super().__init__()
        self.feature_tokenize = feature_tokenize
        self.num_features = num_features
        self.target_feature_index = pollution_index
        self.use_global_token = use_global_token
        if feature_tokenize:
            self.value_projection = nn.Linear(1, d_model)
            self.feature_embedding = nn.Embedding(num_features, d_model)
            if use_global_token:
                self.global_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.input_projection = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.decoder = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, src: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.feature_tokenize:
            batch_size, seq_len, num_features = src.shape
            values = src.view(batch_size, seq_len * num_features, 1)
            value_proj = self.value_projection(values)
            feature_ids = torch.arange(num_features, device=src.device).repeat(seq_len)
            feat_embed = self.feature_embedding(feature_ids).unsqueeze(0).expand(batch_size, -1, -1)
            x = value_proj + feat_embed
            if self.use_global_token:
                global_token = self.global_token.expand(batch_size, -1, -1)
                x = torch.cat([x, global_token], dim=1)
        else:
            x = self.input_projection(src)
        x = self.positional_encoding(x)
        encoded = self.encoder(x, mask=attn_mask)
        if self.feature_tokenize:
            if self.use_global_token:
                summary_vector = encoded[:, -1, :]
            else:
                seq_len = src.size(1)
                target_idx = (seq_len - 1) * self.num_features + self.target_feature_index
                summary_vector = encoded[:, target_idx, :]
        else:
            summary_vector = encoded[:, -1, :]
        output = self.decoder(summary_vector)
        return output.squeeze(-1)


# ---------------------------------------------------------------------------
# Training & evaluation routines
# ---------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float = 0.0,
    attn_mask: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    running_loss = 0.0
    for features, targets in loader:
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(features, attn_mask=attn_mask)
        loss = criterion(outputs, targets)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    return running_loss / len(loader.dataset)


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler,
    target_idx: int,
    attn_mask: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for features, targets in loader:
            outputs = model(features.to(DEVICE), attn_mask=attn_mask)
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
    parser.add_argument("--patience", type=int, default=5, help="Epochs to wait for validation RMSE improvement before stopping.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum RMSE improvement to reset patience.")
    parser.add_argument("--log-every", type=int, default=5, help="How often to log progress (epochs).")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient-norm clipping value (0 disables clipping).")
    parser.add_argument(
        "--feature-tokenize",
        action="store_true",
        help="Represent each (time, feature) pair as its own token and apply feature-level causal masking.",
    )
    parser.add_argument(
        "--disable-global-token",
        action="store_true",
        help="When using feature tokenization, do not append a learnable global token for pooling.",
    )
    parser.add_argument(
        "--causal-adj-path",
        type=Path,
        default=DEFAULT_CAUSAL_ADJ_PATH,
        help="Path to the lagged causal adjacency CSV used to derive attention masks.",
    )
    parser.add_argument(
        "--attention-max-lag",
        type=int,
        default=None,
        help="Override for the maximum temporal lag (in steps) that attention is allowed to access."
        " If unset, inferred from the causal adjacency graph if available.",
    )
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

    attention_max_lag = args.attention_max_lag
    if attention_max_lag is None:
        attention_max_lag = infer_max_lag_from_adjacency(args.causal_adj_path)

    feature_edges = extract_feature_edges(args.causal_adj_path)
    use_global_token = args.feature_tokenize and not args.disable_global_token

    attention_mask: Optional[torch.Tensor]
    if args.feature_tokenize:
        mask, _ = build_feature_token_attention_mask(
            seq_len=args.input_window,
            num_features=len(NUMERIC_COLUMNS),
            max_lag=attention_max_lag,
            edges=feature_edges,
            add_global_token=use_global_token,
        )
        attention_mask = mask.to(DEVICE)
    elif attention_max_lag is not None:
        attention_mask = build_causal_attention_mask(args.input_window, attention_max_lag).to(DEVICE)
    else:
        attention_mask = None

    model = TimeSeriesTransformer(
        num_features=len(NUMERIC_COLUMNS),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout,
        feature_tokenize=args.feature_tokenize,
        pollution_index=NUMERIC_COLUMNS.index("pollution"),
        use_global_token=use_global_token,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_state = None
    best_epoch = 0
    best_val_rmse = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            grad_clip=args.grad_clip,
            attn_mask=attention_mask,
        )
        val_rmse, val_mae = evaluate_metrics(
            model,
            test_loader,
            scaler,
            target_idx,
            attn_mask=attention_mask,
        )

        improved = val_rmse + args.min_delta < best_val_rmse
        if improved:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % args.log_every == 0 or improved or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d}: train_loss={train_loss:.4f} | "
                f"val_RMSE={val_rmse:.3f} | val_MAE={val_mae:.3f} | "
                f"best_RMSE={best_val_rmse:.3f} @ epoch {best_epoch}",
                flush=True,
            )

        if patience_counter >= args.patience:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(no RMSE improvement for {args.patience} epochs).",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_rmse, train_mae = evaluate_metrics(
        model,
        train_loader,
        scaler,
        target_idx,
        attn_mask=attention_mask,
    )
    test_rmse, test_mae = evaluate_metrics(
        model,
        test_loader,
        scaler,
        target_idx,
        attn_mask=attention_mask,
    )

    print("\nFinal evaluation:")
    print(f"Train RMSE: {train_rmse:.3f}\tTrain MAE: {train_mae:.3f}")
    print(f"Test RMSE:  {test_rmse:.3f}\tTest MAE:  {test_mae:.3f}")


if __name__ == "__main__":
    main()
