<div align="center">

# Streamflow-Net

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

## Description

**Author:** Basil Kraft

Streamflow-Net is a data-driven neural network architecture for modeling and routing streamflow in distributed hydrological systems.

Designed to capture spatial and temporal dynamics of runoff generation and flow routing, Streamflow-Net leverages deep learning to simulate hydrological behavior across catchments with flexible, modular components.

This repository contains the code accompanying the paper:

> **TODO:** authors, title, journal, year, DOI.

## Project Structure

```bash
.
├── project/                      # Installable package
│   ├── core/                     # LightningCLI wrapper and Optuna-based tuning/xval engine
│   ├── nn/                       # Model definitions (RunoffNet, StreamflowNet, layers, sub-models)
│   ├── data/                     # LightningDataModule, dataset/graph construction and data splits
│   ├── scripts/                  # Data download/preprocessing and post-processing scripts
│   └── utils/                    # Paths, metrics, plotting, normalization and other shared utilities
├── experiments/                  # LightningCLI configs
│   ├── paths.example.yaml        # Template for your local paths.yaml (data/log directories)
│   ├── base_config.yaml          # Shared defaults
│   ├── fast_dev_run_config.yaml  # Tiny, fast setting for checking the setup
│   ├── prevah/config.yaml        # Runoff model pretrained on PREVAH simulations
│   └── {lumped,routed,instant}/
│       ├── config.yaml           # Model-specific config
│       └── {easy,hard,random}/config.yaml  # Split-specific overrides
├── analysis/                     # Notebooks and scripts for the figures, and the figures themselves
├── slurm/                        # SLURM job scripts; cluster.example.sh is the template for your cluster.sh
├── tests/                        # Pytest suite
├── tune.py                       # Entry point: hyperparameter tuning
├── xval.py                       # Entry point: cross-validation
├── run_all.sh                    # Submits the full tune+xval job graph across experiments/splits
├── pyproject.toml                # Package metadata and pixi environment/dependencies
└── pixi.lock                     # Locked environment
```

## Requirements

- Linux (x86-64); the environment is locked for `linux-64` only.
- An NVIDIA GPU for training. PyTorch is installed with CUDA 12.6, so a recent NVIDIA driver is needed. The tests run on CPU.

## Setup

This project uses [pixi](https://pixi.sh) for environment management.

```bash
git clone <this-repo-url>
cd streamflow-net
pixi install
pixi run setup  # optional: install the git hooks (ruff, nbstripout)
```

### Data

All experiments read the harmonized HRU dataset `harmonized_hru_data.zarr`.

> **TODO:** where to download it (data repository, DOI).

Alternatively, it can be rebuilt from the raw data with `project/scripts/download_data.py` and `project/scripts/preprocessing.py`, which need the raw data paths in `experiments/paths.yaml`.

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

Experiment configs derive their paths from it via `${data_root}`/`${log_root}` interpolation; scripts, plotting and notebooks read it through `project/utils/paths.py`. The remaining keys are only needed to rebuild the dataset from raw data or for the analysis notebooks.

### SLURM

The cluster-specific SLURM settings (account, GPU type, modules) live in `slurm/cluster.sh`, which is not tracked by git either. Create it from the template:

```bash
cp slurm/cluster.example.sh slurm/cluster.sh
```

All scripts load it automatically. Run them with `bash` from the repository root; the job scripts in `slurm/` submit themselves.

## Usage

### Experiments

| Experiment | Model | Description |
| --- | --- | --- |
| `prevah` | RunoffNet | Runoff model pretrained on PREVAH simulations, at HRU level. Required by `routed` and `instant`. |
| `lumped` | RunoffNet | LSTM baseline on catchment-aggregated inputs. |
| `routed` | StreamflowNet | Pretrained runoff model coupled with learned routing along the river network. |
| `instant` | StreamflowNet | As `routed`, but with instantaneous routing (area-weighted summation), as an ablation. |

`lumped`, `routed` and `instant` are run for three data splits: `easy` and `hard` hold out fixed sets of catchments for testing, `random` splits at random (see `project/data/splitting.py`).

### Running

Each experiment runs hyperparameter tuning first, then cross-validation with the best hyperparameters. `routed` and `instant` load the pretrained runoff model from `<log_root>/prevah`, so `prevah` has to be tuned before them.

```bash
# 1. Pretrain the runoff model (no splits).
pixi run python tune.py -c experiments/paths.yaml -c experiments/base_config.yaml -c experiments/prevah/config.yaml
pixi run python xval.py -c experiments/paths.yaml -c experiments/base_config.yaml -c experiments/prevah/config.yaml

# 2. Any other experiment, for <model> in lumped, routed, instant and <split> in easy, hard, random.
pixi run python tune.py -c experiments/paths.yaml -c experiments/base_config.yaml \
    -c experiments/<model>/config.yaml -c experiments/<model>/<split>/config.yaml
pixi run python xval.py -c experiments/paths.yaml -c experiments/base_config.yaml \
    -c experiments/<model>/config.yaml -c experiments/<model>/<split>/config.yaml
```

Adding `-c experiments/fast_dev_run_config.yaml` runs a tiny version (2 trials, 1 epoch, short sequences) to check the setup.

On a SLURM cluster, `bash run_all.sh` submits all experiments and splits in the right order, with several tuning workers per experiment (see [SLURM](#slurm) for the one-time setup). To sanity-check your setup with a single fast, low-resource job first:

```bash
bash slurm/example_minimal.sh
```

### Results and figures

After the runs, `bash slurm/combine_predictions.sh` combines the cross-validation predictions of all experiments into `<log_root>/predictions_combined.zarr`. The notebooks in `analysis/` produce the figures in `analysis/figures/`.

> **TODO:** which notebook produces which figure of the paper.

Some plotting utilities render via Plotly and require a local Chrome install:

```bash
pixi run plotly_get_chrome
```

## Tests

```bash
pixi run pytest
```

The tests run on small synthetic inputs and need neither the dataset nor a GPU. They cover:

- **Routing:** mass conservation on synthetic river networks, instant routing, time sub-steps, graph construction and path sampling.
- **Runoff model:** chunked inference matches a single pass in outputs and gradients.
- **Loss, metrics and normalization:** analytically known values (e.g. NSE, KGE), NaN masking and exact inverse transforms.
- **Data splits and unit conversions.**
- **Configuration:** every experiment config resolves through the CLI, and no machine-specific paths are tracked outside `experiments/paths.yaml`.

Two tests use the real dataset at `data_root` and are skipped automatically if it isn't found.

## Citation

> **TODO:** BibTeX entry for the paper.

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE).
