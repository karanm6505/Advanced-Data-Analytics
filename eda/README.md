# Exploratory Data Analysis Toolkit

This directory houses the assets that generate a common set of exploratory data analysis (EDA) artefacts for every CSV in the repository. The goal is to give you consistent quick-look diagnostics that can feed into the causal inference and deep learning workflow outlined in the project brief.

## What the script does

Executing the notebook `eda/eda_analysis.ipynb` creates a per-dataset package of:

- **Structured metadata** (`*_summary.json`) capturing basic stats, stationarity diagnostics (ADF), inferred frequency, and missing-value breakdowns.
- **Figures** under `reports/figures/<dataset-stem>/` including:
  - Raw series plots and empirical distributions.
  - Rolling mean/variance overlays for drift inspection.
  - Seasonal decomposition (when enough history exists).
  - Autocorrelation and partial autocorrelation diagnostics.
  - For the multivariate pollution data: correlation heatmaps and categorical wind-direction counts.

These artefacts provide the factual backbone for Phase 1 baselining and inform feature engineering prior to the causal discovery phase.

## Usage

Create and activate a Python environment (>=3.9) using [uv](https://github.com/astral-sh/uv):

```bash
uv venv
source .venv/bin/activate
uv pip install pandas numpy matplotlib seaborn statsmodels
```

Then open and execute the notebook:

1. `uv pip install notebook` (or launch Jupyter via your preferred IDE).
2. Run every cell in `eda/eda_analysis.ipynb`.

The notebook prints progress for each dataset and writes outputs into `eda/reports/`. Re-running it will refresh the artefacts.

## Adapting the workflow

- To add a new dataset, extend the `DATASETS` dictionary in `run_eda.py` with the appropriate metadata (date parsing callable, target column, etc.).
- If you need extra diagnostics (e.g., KPSS tests, spectral density plots), implement them in a helper and call the function within `run_univariate_workflow` or `run_multivariate_workflow`.
- The JSON summaries are intentionally machine-readable so you can ingest them into notebooks or dashboards downstream.
