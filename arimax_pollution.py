"""Train an ARIMAX model on the multivariate pollution dataset and
evaluate it on both an internal hold-out window and an external
pollution_test_data1.csv sample."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

REPO_ROOT = Path(__file__).resolve().parent
TRAIN_CSV = REPO_ROOT / "LSTM-Multivariate_pollution.csv"
TEST_CSV = REPO_ROOT / "pollution_test_data1.csv"
TARGET_COLUMN = "pollution"
EXOG_COLUMNS = ["dew", "temp", "press", "wnd_spd", "snow", "rain"]
AUTO_ARIMA_MAX_SAMPLES = 3000


def load_training_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df[EXOG_COLUMNS] = df[EXOG_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    df = df.loc[df[EXOG_COLUMNS].notnull().all(axis=1)]
    if len(df) > 5000:
        df = df.iloc[-5000:]
    return df


def load_external_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df[EXOG_COLUMNS] = df[EXOG_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    df = df.loc[df[EXOG_COLUMNS].notnull().all(axis=1)]
    df.reset_index(drop=True, inplace=True)
    df.index = pd.RangeIndex(start=0, stop=len(df), step=1)
    return df


def scale_exog(
    df: pd.DataFrame,
    columns: list[str],
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[columns])
    else:
        scaled = scaler.transform(df[columns])
    exog_df = pd.DataFrame(scaled, columns=columns, index=df.index)
    return exog_df, scaler


def train_validation_split(
    target: pd.Series, exog: pd.DataFrame, train_ratio: float = 0.8
):
    split_idx = int(len(target) * train_ratio)
    y_train = target.iloc[:split_idx]
    y_val = target.iloc[split_idx:]
    x_train = exog.iloc[:split_idx]
    x_val = exog.iloc[split_idx:]
    return y_train, y_val, x_train, x_val


def select_order(endog, exog) -> tuple[int, int, int]:
    if len(endog) > AUTO_ARIMA_MAX_SAMPLES:
        endog = endog.iloc[-AUTO_ARIMA_MAX_SAMPLES:]
        exog = exog.iloc[-AUTO_ARIMA_MAX_SAMPLES:]

    auto_model = auto_arima(
        endog,
        exogenous=exog,
        start_p=1,
        start_q=1,
        max_p=2,
        max_q=2,
        seasonal=False,
        d=1,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    order = auto_model.order
    return order


def build_model(endog, exog, order):
    return ARIMA(endog, order=order, exog=exog, enforce_stationarity=False, enforce_invertibility=False)


def evaluate(true_values, predictions) -> tuple[float, float]:
    mae = mean_absolute_error(true_values, predictions)
    rmse = float(np.sqrt(mean_squared_error(true_values, predictions)))
    return mae, rmse


def main() -> None:
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError(
            "Training or test CSV missing. Ensure both pollution datasets are in the repo root."
        )

    train_df = load_training_frame(TRAIN_CSV)
    exog_df, scaler = scale_exog(train_df, EXOG_COLUMNS)

    target_series = train_df[TARGET_COLUMN]
    log_target = np.log1p(target_series)
    y_train_log, y_val_log, x_train, x_val = train_validation_split(
        log_target, exog_df
    )
    split_idx = len(y_train_log)
    y_val_actual = target_series.iloc[split_idx:]

    order = select_order(y_train_log, x_train)
    model = build_model(y_train_log, x_train, order)
    result = model.fit()

    val_forecast = result.get_forecast(steps=len(y_val_log), exog=x_val)
    val_pred_log = val_forecast.predicted_mean
    val_pred = np.expm1(val_pred_log)
    val_mae, val_rmse = evaluate(y_val_actual, val_pred)

    external_df = load_external_frame(TEST_CSV)
    external_exog, _ = scale_exog(external_df, EXOG_COLUMNS, scaler=scaler)
    external_forecast = result.get_forecast(
        steps=len(external_df), exog=external_exog
    )
    external_pred_log = external_forecast.predicted_mean
    external_pred = np.expm1(external_pred_log)
    ext_mae, ext_rmse = evaluate(external_df[TARGET_COLUMN], external_pred)

    print("=== ARIMAX Pollution Forecast ===")
    print(f"Training samples used: {len(y_train_log)}")
    print(f"Validation samples: {len(y_val_log)}")
    print(f"External test samples: {len(external_df)}")
    print(f"Selected order: {order}")
    print("-- Hold-out window --")
    print(f"MAE: {val_mae:.2f}")
    print(f"RMSE: {val_rmse:.2f}")
    print("-- External pollution_test_data1.csv --")
    print(f"MAE: {ext_mae:.2f}")
    print(f"RMSE: {ext_rmse:.2f}")


if __name__ == "__main__":
    main()