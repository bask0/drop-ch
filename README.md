<div align="center">

# Streamflow-Net

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

## Description

**Author:** Basil Kraft

Streamflow-Net is a data-driven neural network architecture for modeling and routing streamflow in distributed hydrological systems.

Designed to capture spatial and temporal dynamics of runoff generation and flow routing, Streamflow-Net leverages deep learning to simulate hydrological behavior across catchments with flexible, modular components.

## Project Structure

```bash
.
├── project/                      # Installable package
│   ├── core/                     # LightningCLI wrapper and Optuna-based tuning/xval engine
│   ├── nn/                       # Model definitions (RunoffNet, StreamflowNet, layers, sub-models)
│   ├── data/                     # LightningDataModule and dataset/graph construction
│   ├── scripts/                  # Standalone data download/preprocessing/analysis scripts
│   └── utils/                    # Metrics, plotting, normalization, and other shared utilities
├── experiments/                  # Per-experiment Hydra/LightningCLI configs
│   ├── paths.example.yaml        # Template for your local paths.yaml (data/log directories)
│   ├── base_config.yaml          # Shared defaults
│   └── {instant,lumped,prevah,routed}/
│       ├── config.yaml           # Model-specific config
│       └── {easy,hard,random}/config.yaml  # Split-specific overrides
├── slurm/                        # SLURM job scripts; cluster.example.sh is the template for your cluster.sh
├── tests/                        # Pytest suite
├── tune.py                       # Entry point: hyperparameter tuning
├── xval.py                       # Entry point: cross-validation
├── run_all.sh                    # Submits the full tune+xval job graph across experiments/splits
├── pyproject.toml                # Package metadata and pixi environment/dependencies
└── pixi.lock                     # Locked environment
```

## Setup

This project uses [pixi](https://pixi.sh) for environment management.

```bash
git clone <this-repo-url>
cd streamflow-net
pixi install
pixi run setup  # optional: install the git hooks (ruff, nbstripout)
```

### Paths

All machine-specific paths live in `experiments/paths.yaml`, which is not tracked by git. Create it from the template and fill in your directories:

```bash
cp experiments/paths.example.yaml experiments/paths.yaml
```

```yaml
data_root: /path/to/data          # directory containing harmonized_hru_data.zarr
log_root: /path/to/logs           # experiment outputs (checkpoints, Optuna databases, logs, predictions)
hru_file: /path/to/hrus.gpkg      # HRU polygons, used for plotting
```

Experiment configs derive their paths from it via `${data_root}`/`${log_root}` interpolation; scripts, plotting and notebooks read it through `project/utils/paths.py`. The remaining keys are only needed to rebuild the dataset from raw data (`project/scripts/download_data.py`, `project/scripts/preprocessing.py`) or for the analysis notebooks.

### SLURM

The cluster-specific SLURM settings (account, GPU type, modules) live in `slurm/cluster.sh`, which is not tracked by git either. Create it from the template:

```bash
cp slurm/cluster.example.sh slurm/cluster.sh
```

All scripts load it automatically. Run them with `bash` from the repository root; the job scripts in `slurm/` submit themselves.

## Usage

Run hyperparameter tuning:

```bash
pixi run python tune.py -c experiments/paths.yaml -c experiments/base_config.yaml -c experiments/<model>/config.yaml -c experiments/<model>/<split>/config.yaml
```

Run cross-validation on the tuned model:

```bash
pixi run python xval.py -c experiments/paths.yaml -c experiments/base_config.yaml -c experiments/<model>/config.yaml -c experiments/<model>/<split>/config.yaml
```

where `<model>` is one of `instant`, `lumped`, `prevah`, `routed` and `<split>` is one of `easy`, `hard`, `random`.

On a SLURM cluster, `bash run_all.sh` submits the full tune+xval job graph across all experiments and splits (see [SLURM](#slurm) for the one-time setup). To sanity-check your setup with a single fast, low-resource job before committing to the full run:

```bash
bash slurm/example_minimal.sh
```

### Plotting

Some plotting utilities render via Plotly and require a local Chrome install:

```bash
pixi run plotly_get_chrome
```

## Tests

```bash
pixi run pytest
```

The tests run on synthetic data and need neither the dataset nor a GPU. Two tests use the real dataset at `data_root` and are skipped automatically if it isn't found.
