"""Run causal discovery on the pollution dataset using the causal-learn PC algorithm."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
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


def load_pollution_frame(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[list(columns)].dropna()
    df = df.iloc[-5000:]
    df["wnd_dir"] = pd.Categorical(df["wnd_dir"]).codes
    return df


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


def main(alpha: float = 0.01) -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "LSTM-Multivariate_pollution.csv not found in repository root."
        )

    df = load_pollution_frame(DATA_PATH, DEFAULT_COLUMNS)
    cg, adjacency = run_pc_causal_discovery(df, alpha=alpha)
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
    adjacency_path = REPO_ROOT / "causal_graph_adjacency.csv"
    adjacency.to_csv(adjacency_path)
    print(f"Adjacency matrix saved to {adjacency_path}")
    plot_causal_graph(adjacency, CAUSAL_FIG_PATH)
    print(f"Causal DAG figure saved to {CAUSAL_FIG_PATH}")


if __name__ == "__main__":
    main()
