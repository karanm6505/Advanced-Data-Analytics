"""Run causal discovery and backdoor adjustment for the pollution dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "LSTM-Multivariate_pollution.csv"
CAUSAL_FIG_PATH = REPO_ROOT / "causal_graph.png"
DEFAULT_COLUMNS = [
    "pollution",
    "dew",
    "temp",
    "press",
    "wnd_spd",
    "snow",
    "rain",
    "wnd_dir",
]
BASE_COLUMNS = ["pollution", "dew", "temp", "press", "wnd_spd", "snow", "rain"]
DEFAULT_ADJUSTED_OUTPUT = REPO_ROOT / "pollution_backdoor_adjusted.csv"
DEFAULT_TEST_PATH = REPO_ROOT / "pollution_test_data1.csv"
DEFAULT_TEST_ADJUSTED_OUTPUT = REPO_ROOT / "pollution_test_backdoor_adjusted.csv"


def load_pollution_frame(path: Path, columns: Iterable[str], keep_date: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    selected_cols = list(columns)
    if keep_date:
        selected_cols.append("date")
    df = df[selected_cols].dropna()
    if "wnd_dir" in df.columns:
        df["wnd_dir"] = pd.Categorical(df["wnd_dir"]).codes
    return df


def _create_lagged_frame(df: pd.DataFrame, lag_steps: int = 1, include_date: bool = True) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    lagged = {}
    if include_date and "date" in df.columns:
        lagged["date"] = df["date"].iloc[lag_steps:].reset_index(drop=True)
    for col in BASE_COLUMNS:
        current = df[col].iloc[lag_steps:].reset_index(drop=True)
        prior = df[col].shift(1).iloc[lag_steps:].reset_index(drop=True)
        lagged[f"{col}_t"] = current
        lagged[f"{col}_t-1"] = prior
    return pd.DataFrame(lagged).dropna().reset_index(drop=True)


def create_lagged_pollution_frame(path: Path, lag_steps: int = 1, include_date: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
        df["date"] = pd.RangeIndex(start=0, stop=len(df), step=1)
    df = df[["date"] + BASE_COLUMNS]
    return _create_lagged_frame(df, lag_steps=lag_steps, include_date=include_date)


def run_pc_causal_discovery(df: pd.DataFrame, alpha: float = 0.01):
    scaler = StandardScaler()
    data = scaler.fit_transform(df.to_numpy())
    cg = pc(data, alpha=alpha, indep_test=fisherz)
    adjacency = pd.DataFrame(
        cg.G.graph.astype(int), index=df.columns, columns=df.columns
    )
    return cg, adjacency


def format_edge_summary(adjacency: pd.DataFrame) -> list[tuple[str, str]]:
    edges = []
    for i, col_i in enumerate(adjacency.columns):
        for j, col_j in enumerate(adjacency.columns):
            if j <= i:
                continue
            if adjacency.iat[i, j] != 0:
                edges.append((col_i, col_j))
    return edges


def plot_causal_graph(adjacency: pd.DataFrame, output_path: Path) -> None:
    nodes = list(adjacency.columns)
    directed = nx.DiGraph()
    undirected = nx.Graph()
    directed.add_nodes_from(nodes)
    undirected.add_nodes_from(nodes)

    for i, src in enumerate(nodes):
        for j, dst in enumerate(nodes):
            if i == j:
                continue
            val = adjacency.iat[i, j]
            if val == 0:
                continue
            reverse = adjacency.iat[j, i]
            if reverse == 0:
                directed.add_edge(src, dst)
            elif i < j:
                undirected.add_edge(src, dst)

    combined = nx.Graph()
    combined.add_nodes_from(nodes)
    combined.add_edges_from(directed.to_undirected().edges())
    combined.add_edges_from(undirected.edges())
    pos = nx.spring_layout(combined, seed=42)

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(combined, pos, node_color="#1f77b4", node_size=1200)
    nx.draw_networkx_labels(combined, pos, font_size=10, font_weight="bold")

    if directed.number_of_edges() > 0:
        nx.draw_networkx_edges(
            directed,
            pos,
            edge_color="#ff7f0e",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            width=2.0,
        )

    if undirected.number_of_edges() > 0:
        nx.draw_networkx_edges(
            undirected,
            pos,
            edge_color="#2ca02c",
            style="dashed",
            arrows=False,
            width=2.0,
        )

    handles = []
    if directed.number_of_edges() > 0:
        handles.append(
            Line2D([0], [0], color="#ff7f0e", lw=2, label="Directed edge")
        )
    if undirected.number_of_edges() > 0:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#2ca02c",
                lw=2,
                linestyle="dashed",
                label="Undirected edge",
            )
        )
    if handles:
        plt.legend(handles=handles, loc="upper left")

    plt.title("Causal Graph (PC algorithm)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parents_of(node: str, adjacency: pd.DataFrame) -> List[str]:
    parents = []
    for candidate in adjacency.columns:
        if adjacency.at[candidate, node] == 1:
            parents.append(candidate)
    return parents


def identify_backdoor_confounders(adjacency: pd.DataFrame, outcome: str, treatments: Sequence[str]) -> List[str]:
    parents_outcome = set(parents_of(outcome, adjacency))
    confounders = set()
    for treatment in treatments:
        parents_treatment = set(parents_of(treatment, adjacency))
        confounders |= parents_outcome & parents_treatment
    lagged_candidates = {col for col in adjacency.columns if col.endswith("_t-1")}
    if not confounders:
        confounders = lagged_candidates
    else:
        confounders |= lagged_candidates
    confounders.discard(outcome)
    for treatment in treatments:
        confounders.discard(treatment)
    return sorted(confounders)


class LinearResidualizer:
    def __init__(self, target_cols: Sequence[str], confounders: Sequence[str]) -> None:
        self.target_cols = list(target_cols)
        self.confounders = list(confounders)
        self.scaler: Optional[StandardScaler] = None
        self.models: dict[str, LinearRegression] = {}
        self.target_means: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "LinearResidualizer":
        if not self.confounders:
            for col in self.target_cols:
                self.target_means[col] = df[col].mean()
            return self

        conf_data = df[self.confounders]
        self.scaler = StandardScaler()
        Z = self.scaler.fit_transform(conf_data)
        for col in self.target_cols:
            y = df[col].to_numpy()
            model = LinearRegression().fit(Z, y)
            self.models[col] = model
            self.target_means[col] = y.mean()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.confounders:
            return df[self.target_cols].copy()
        if self.scaler is None:
            raise RuntimeError("Residualizer must be fit before calling transform().")
        missing = set(self.confounders) - set(df.columns)
        if missing:
            raise ValueError(f"Input frame is missing confounder columns: {sorted(missing)}")
        Z = self.scaler.transform(df[self.confounders])
        adjusted = pd.DataFrame(index=df.index)
        for col in self.target_cols:
            y = df[col].to_numpy()
            pred = self.models[col].predict(Z)
            adjusted[col] = y - pred + self.target_means[col]
        return adjusted


def _compose_adjusted_dataset(
    lagged: pd.DataFrame,
    adjusted_values: pd.DataFrame,
    target_mix: float,
    adjust_features: bool,
    outcome: str,
    include_date: bool,
) -> pd.DataFrame:
    result = pd.DataFrame()
    if include_date and "date" in lagged.columns:
        result["date"] = lagged["date"].reset_index(drop=True)
    for col in BASE_COLUMNS:
        source = f"{col}_t"
        if col == "pollution":
            residual_col = adjusted_values[outcome].reset_index(drop=True)
            result[col] = target_mix * residual_col + (1 - target_mix) * lagged[source].reset_index(drop=True)
        elif adjust_features and source in adjusted_values:
            result[col] = adjusted_values[source].reset_index(drop=True)
        else:
            result[col] = lagged[source].reset_index(drop=True)
    return result


def run_discovery(alpha: float = 0.01, use_lagged: bool = False) -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "LSTM-Multivariate_pollution.csv not found in repository root."
        )

    if use_lagged:
        lagged = create_lagged_pollution_frame(DATA_PATH)
        df = lagged.drop(columns=["date"])
        suffix = "lagged"
    else:
        df = load_pollution_frame(DATA_PATH, DEFAULT_COLUMNS)
        suffix = "base"

    _, adjacency = run_pc_causal_discovery(df, alpha=alpha)
    edges = format_edge_summary(adjacency)

    print("=== Causal Discovery (PC algorithm, causal-learn) ===")
    print(f"Rows analyzed: {len(df)}")
    print(f"Variables: {', '.join(df.columns)}")
    print(f"Significance level (alpha): {alpha}")
    print("-- Undirected/partially directed edges detected --")
    if not edges:
        print("No edges detected at the specified alpha level.")
    else:
        for left, right in edges:
            print(f"{left} -- {right}")

    adjacency_path = REPO_ROOT / f"causal_graph_{suffix}_adjacency.csv"
    adjacency.to_csv(adjacency_path)
    print(f"Adjacency matrix saved to {adjacency_path}")

    plot_path = REPO_ROOT / f"causal_graph_{suffix}.png"
    plot_causal_graph(adjacency, plot_path)
    print(f"Causal DAG figure saved to {plot_path}")

    if not use_lagged:
        adjacency.to_csv(REPO_ROOT / "causal_graph_adjacency.csv")
        plot_causal_graph(adjacency, CAUSAL_FIG_PATH)


def run_backdoor_adjustment(
    alpha: float = 0.01,
    output_path: Path = DEFAULT_ADJUSTED_OUTPUT,
    adjust_features: bool = False,
    target_mix: float = 1.0,
    test_path: Optional[Path] = DEFAULT_TEST_PATH,
    test_output_path: Optional[Path] = DEFAULT_TEST_ADJUSTED_OUTPUT,
    skip_test_adjust: bool = False,
) -> None:
    lagged = create_lagged_pollution_frame(DATA_PATH, include_date=True)
    df = lagged.drop(columns=["date"])
    _, adjacency = run_pc_causal_discovery(df, alpha=alpha)

    treatments = [f"{col}_t" for col in BASE_COLUMNS]
    outcome = "pollution_t"
    confounders = identify_backdoor_confounders(adjacency, outcome, treatments)

    print("=== Backdoor Adjustment ===")
    print(f"Confounders used ({len(confounders)}): {', '.join(confounders)}")

    to_adjust = [outcome] if not adjust_features else treatments + [outcome]
    target_mix = float(np.clip(target_mix, 0.0, 1.0))
    residualizer = LinearResidualizer(to_adjust, confounders)
    residualizer.fit(lagged)
    adjusted_values = residualizer.transform(lagged)

    adjusted = _compose_adjusted_dataset(
        lagged,
        adjusted_values,
        target_mix=target_mix,
        adjust_features=adjust_features,
        outcome=outcome,
        include_date=True,
    )

    adjusted.to_csv(output_path, index=False)
    print(f"Causally adjusted dataset saved to {output_path}")

    if skip_test_adjust or test_path is None or test_output_path is None:
        return

    try:
        test_lagged = create_lagged_pollution_frame(test_path, include_date=False)
    except FileNotFoundError:
        print(f"Test file {test_path} not found; skipping test adjustment.")
        return

    missing_confounders = set(confounders) - set(test_lagged.columns)
    if missing_confounders:
        print(
            "Cannot adjust test set because it lacks confounder columns: "
            f"{sorted(missing_confounders)}"
        )
        return

    test_adjusted_values = residualizer.transform(test_lagged)
    adjusted_test = _compose_adjusted_dataset(
        test_lagged,
        test_adjusted_values,
        target_mix=target_mix,
        adjust_features=adjust_features,
        outcome=outcome,
        include_date=False,
    )
    adjusted_test.to_csv(test_output_path, index=False)
    print(f"Adjusted test dataset saved to {test_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal discovery and backdoor adjustment workflows.")
    parser.add_argument("--action", choices=["discover", "adjust", "all"], default="discover", help="Which pipeline to run.")
    parser.add_argument("--alpha", type=float, default=0.01, help="Significance level for Fisher's Z test.")
    parser.add_argument("--lagged", action="store_true", help="Use lagged features for discovery.")
    parser.add_argument("--output", type=Path, default=DEFAULT_ADJUSTED_OUTPUT, help="Path for the adjusted dataset.")
    parser.add_argument(
        "--test-path",
        type=Path,
        default=DEFAULT_TEST_PATH,
        help="Optional path to the held-out test CSV to adjust alongside the training data.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=DEFAULT_TEST_ADJUSTED_OUTPUT,
        help="Where to write the adjusted test CSV (only used when test-path is provided).",
    )
    parser.add_argument(
        "--skip-test-adjust",
        action="store_true",
        help="Skip generating an adjusted test CSV even if a test path is available.",
    )
    parser.add_argument(
        "--adjust-features",
        action="store_true",
        help="Residualize all treatments instead of only the target column.",
    )
    parser.add_argument(
        "--target-mix",
        type=float,
        default=1.0,
        help="Blend factor between original target (0) and fully residualized target (1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action in {"discover", "all"}:
        run_discovery(alpha=args.alpha, use_lagged=args.lagged)
    if args.action in {"adjust", "all"}:
        run_backdoor_adjustment(
            alpha=args.alpha,
            output_path=args.output,
            adjust_features=args.adjust_features,
            target_mix=args.target_mix,
            test_path=None if args.skip_test_adjust else args.test_path,
            test_output_path=None if args.skip_test_adjust else args.test_output,
            skip_test_adjust=args.skip_test_adjust,
        )


if __name__ == "__main__":
    main()
