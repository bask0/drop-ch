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
│   ├── paths.yaml                 # Your data/log root directories (edit this)
│   ├── base_config.yaml          # Shared defaults
│   └── {instant,lumped,prevah,routed}/
│       ├── config.yaml           # Model-specific config
│       └── {easy,hard,random}/config.yaml  # Split-specific overrides
├── slurm/                        # SLURM job scripts (written for a Slurm + pixi cluster setup)
├── tests/                        # Pytest suite
├── tune.py                       # Entry point: hyperparameter tuning
├── xval.py                       # Entry point: cross-validation
├── run_all.sh                    # Submits the full tune+xval job graph across experiments/splits
├── pyproject.toml                # Package metadata and pixi environment/dependencies
└── pixi.lock                     # Locked environment
```

All experiment configs derive their data and output paths from `experiments/paths.yaml` via `${data_root}`/`${log_root}` interpolation — it's the only file you need to edit to run on your own machine:

```yaml
data_root: /path/to/data   # directory containing harmonized_hru_data.zarr
log_root: /path/to/logs    # directory for experiment outputs (checkpoints, optuna DBs, logs)
```

## Setup

This project uses [pixi](https://pixi.sh) for environment management.

```bash
git clone <this-repo-url>
cd streamflow-net
pixi install
```

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

On a SLURM cluster, `run_all.sh` submits the full tune+xval job graph across all experiments and splits (adjust `slurm/*.sh` for your cluster's account/module setup first). To sanity-check your setup with a single fast, low-resource job before committing to the full run:

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

The smoke test in `tests/test_pipeline.py` requires the processed HRU dataset (path configured at the top of the file) and is skipped automatically if it isn't found.
