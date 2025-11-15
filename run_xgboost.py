import csv
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import xgboost as xgb

BASE_DIR = pathlib.Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "LSTM-Multivariate_pollution.csv"
TEST_PATH = BASE_DIR / "pollution_test_data1.csv"

NUMERIC_KEYS = ["dew", "temp", "press", "wnd_spd", "snow", "rain"]
CATEGORICAL_KEY = "wnd_dir"

RANDOM_SEED = 67
np.random.seed(RANDOM_SEED)


def _cast_numeric(value: str) -> float:
    """Convert string inputs to float while handling empty markers."""
    if value is None:
        return float("nan")
    value_str = value.strip()
    if value_str == "" or value_str.lower() in {"na", "nan"}:
        return float("nan")
    return float(value_str)


def load_dataset(path: pathlib.Path) -> Tuple[List[Dict[str, float]], List[float], List[Dict[str, float]]]:
    """Parse CSV rows into feature dictionaries, target values, and raw records."""
    features: List[Dict[str, float]] = []
    targets: List[float] = []
    raw_records: List[Dict[str, float]] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target_raw = row.get("pollution")
            if target_raw is None:
                continue

            target_val = _cast_numeric(target_raw)
            if np.isnan(target_val):
                continue

            feature_row: Dict[str, float] = {}
            record_row: Dict[str, float] = {}

            for key, value in row.items():
                if key in {"pollution", "date"}:
                    continue
                if key == CATEGORICAL_KEY:
                    safe_val = (value or "UNK").strip() or "UNK"
                    feature_row[key] = safe_val
                    record_row[key] = safe_val
                else:
                    casted = _cast_numeric(value)
                    feature_row[key] = casted
                    record_row[key] = casted

            features.append(feature_row)
            targets.append(target_val)
            record_row["pollution"] = target_val
            raw_records.append(record_row)

    return features, targets, raw_records


def build_category_mapping(feature_rows: List[Dict[str, float]]) -> Dict[str, int]:
    """Generate a stable mapping for categorical wind direction values."""
    categories = {
        (row.get(CATEGORICAL_KEY) or "UNK").strip() or "UNK"
        for row in feature_rows
    }
    categories.discard("UNK")
    ordered = ["UNK"] + sorted(categories)
    return {category: idx for idx, category in enumerate(ordered)}


def build_design_matrix(
    feature_rows: List[Dict[str, float]],
    cat_mapping: Dict[str, int],
    numeric_fill: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert feature dictionaries into a numeric matrix with one-hot wind dirs."""
    n_samples = len(feature_rows)
    n_numeric = len(NUMERIC_KEYS)
    n_categories = len(cat_mapping)

    matrix = np.zeros((n_samples, n_numeric + n_categories), dtype=np.float32)
    matrix[:, :n_numeric] = np.nan

    for row_index, row in enumerate(feature_rows):
        for col_index, key in enumerate(NUMERIC_KEYS):
            matrix[row_index, col_index] = row.get(key, float("nan"))

        category = (row.get(CATEGORICAL_KEY) or "UNK").strip() or "UNK"
        category_index = cat_mapping.get(category, cat_mapping["UNK"])
        matrix[row_index, n_numeric + category_index] = 1.0

    if numeric_fill is None:
        fill_values = np.nanmedian(matrix[:, :n_numeric], axis=0)
        fill_values = np.where(np.isnan(fill_values), 0.0, fill_values)
    else:
        fill_values = numeric_fill

    for col_index in range(n_numeric):
        column = matrix[:, col_index]
        missing_mask = np.isnan(column)
        if missing_mask.any():
            column[missing_mask] = fill_values[col_index]
            matrix[:, col_index] = column

    return matrix, fill_values


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics without relying on external libraries."""
    residual = y_pred - y_true
    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    ss_res = float(np.sum(residual ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_predictions(
    input_rows: List[Dict[str, float]],
    predictions: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    """Persist predictions alongside their source rows for inspection."""
    if not input_rows:
        return

    fieldnames = list(input_rows[0].keys())
    if "pollution_pred" not in fieldnames:
        fieldnames.append("pollution_pred")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, pred in zip(input_rows, predictions):
            row_out = row.copy()
            row_out["pollution_pred"] = f"{pred:.4f}"
            writer.writerow(row_out)


def train_xgboost(dtrain: xgb.DMatrix, dvalid: xgb.DMatrix) -> xgb.Booster:
    """Train an XGBoost regressor using the low-level training API."""
    params = {
        "objective": "reg:squarederror",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "seed": RANDOM_SEED,
        "verbosity": 1,
    }

    evals = [(dtrain, "train"), (dvalid, "valid")]
    return xgb.train(
        params,
        dtrain,
        num_boost_round=600,
        evals=evals,
        verbose_eval=100,
    )


def main() -> None:
    feature_rows, targets, raw_records = load_dataset(TRAIN_PATH)

    if not feature_rows:
        raise RuntimeError("Training dataset produced no usable rows; check source file.")

    category_mapping = build_category_mapping(feature_rows)
    design_matrix, fill_values = build_design_matrix(feature_rows, category_mapping)
    target_array = np.asarray(targets, dtype=np.float32)

    split_index = int(len(design_matrix) * 0.8)
    split_index = max(1, min(split_index, len(design_matrix) - 1))

    X_train = design_matrix[:split_index]
    y_train = target_array[:split_index]
    X_valid = design_matrix[split_index:]
    y_valid = target_array[split_index:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    booster = train_xgboost(dtrain, dvalid)

    valid_predictions = booster.predict(dvalid)
    validation_metrics = evaluate_predictions(y_valid, valid_predictions)

    print("Validation metrics:")
    for name, value in validation_metrics.items():
        print(f"  {name.upper()}: {value:.4f}")

    if TEST_PATH.exists():
        test_features, test_targets, test_records = load_dataset(TEST_PATH)
        test_matrix, _ = build_design_matrix(test_features, category_mapping, fill_values)
        dtest = xgb.DMatrix(test_matrix)
        test_predictions = booster.predict(dtest)

        output_path = BASE_DIR / "xgboost_predictions.csv"
        save_predictions(test_records, test_predictions, output_path)
        print(f"Saved predictions to {output_path}")

        if test_targets:
            test_metrics = evaluate_predictions(
                np.asarray(test_targets, dtype=np.float32),
                test_predictions,
            )
            print("Test file metrics (target column detected):")
            for name, value in test_metrics.items():
                print(f"  {name.upper()}: {value:.4f}")


if __name__ == "__main__":
    main()
