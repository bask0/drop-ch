<div align="center">

# DROP

[![Tests](https://github.com/bask0/drop-ch/actions/workflows/tests.yml/badge.svg)](https://github.com/bask0/drop-ch/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
<!-- Once the paper is published, uncomment and add the links:
[![Paper](https://img.shields.io/badge/Paper-JOURNAL%20(YEAR)-blue.svg)](https://doi.org/PAPER_DOI)
[![Data](https://img.shields.io/badge/Data-Dataset-green.svg)](https://doi.org/DATA_DOI)
-->

<!-- Once the paper is published, uncomment and add a figure from the paper:
<figure>
  <img src="./analysis/figures/FIGURE.png" alt="DROP" style="width:70%">
  <figcaption></br>Fig. N from AUTHORS (YEAR): CAPTION.</figcaption>
</figure>
-->

</div>

## Description

DROP is a data-driven neural network architecture for modeling and routing streamflow in distributed hydrological systems.

Designed to capture spatial and temporal dynamics of runoff generation and flow routing, DROP leverages deep learning to simulate hydrological behavior across catchments with flexible, modular components. In the code, the model is the class `DROP` in `project/nn/drop_model.py`.

This repository contains the code accompanying a paper on DROP.

> [!NOTE]
> If your published work uses or builds on this code, please cite the paper (see [Citation](#citation)). The reference will be added once the paper is published.

This repository is shared to make the methods and results of the paper transparent and reproducible. It is research code, organized around the experiments in the paper, rather than a general-purpose tool for streamflow modeling. It was developed for our data and compute environment, so running the experiments and notebooks elsewhere will require some adaptation, and applying the model to other regions or datasets even more so.

## Project Structure

```bash
.
├── project/                      # Installable package
│   ├── core/                     # LightningCLI wrapper and Optuna-based tuning/xval engine
│   ├── nn/                       # Model definitions (RunoffNet, DROP, layers, sub-models)
│   ├── data/                     # LightningDataModule, dataset/graph construction and data splits
│   ├── scripts/                  # Data download/preprocessing and post-processing scripts
│   └── utils/                    # Paths, metrics, plotting, normalization and other shared utilities
├── experiments/                  # LightningCLI configs
│   ├── paths.example.yaml        # Template for your local paths.yaml (data/log directories)
│   ├── base_config.yaml          # Shared defaults
│   ├── fast_dev_run_config.yaml  # Tiny, fast setting for checking the setup
│   ├── prevah/config.yaml        # Runoff model pretrained on PREVAH simulations
│   ├── prevah/run.sh             # SLURM submit script for this experiment
│   └── {lumped,routed,instant}/
│       ├── config.yaml           # Model-specific config
│       ├── run.sh                # SLURM submit script for this experiment
│       └── {easy,hard,random}/config.yaml  # Split-specific overrides
├── analysis/                     # Notebooks and scripts for the figures, and the figures themselves
├── slurm/                        # SLURM job scripts; cluster.example.sh is the template for your cluster.sh
├── tests/                        # Pytest suite
├── .github/workflows/            # CI: ruff and tests on every push to main and every pull request
├── tune.py                       # Entry point: hyperparameter tuning
├── xval.py                       # Entry point: predictions of the best tuned model
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
git clone https://github.com/bask0/drop-ch.git
cd drop-ch
pixi install
pixi run setup  # optional: install the git hooks (ruff, nbstripout)
```

### Data

All experiments read the harmonized HRU dataset `harmonized_hru_data.zarr`.

> [!NOTE]
> The download location of the dataset will be added here once the paper is published.

Alternatively, it can be rebuilt from the raw data with `project/scripts/download_data.py` and `project/scripts/preprocessing.py`, which need the raw data paths in `experiments/paths.yaml`.

The dataset is built from third-party data, which is not part of this repository and remains subject to the terms of its providers. Check those terms before using or redistributing the data:

- **Meteorological forcing:** MeteoSwiss gridded precipitation and temperature (RhiresD, TabsD) and ERA5-Land.
- **Simulated runoff:** PREVAH.
- **Observed discharge:** FOEN and, for gauges outside Switzerland, other sources.
- **Catchment attributes:** MERIT Hydro, SoilGrids, CORINE land cover, a dam inventory and a river network.

The analysis notebooks additionally use CAMELS-CH and draw the EU-Hydro river network on the maps; set `river_file` in `experiments/paths.yaml` to the river network GeoPackage.

### Paths

All machine-specific paths live in `experiments/paths.yaml`, which is not tracked by git. Create it from the template and fill in your directories:

```bash
cp experiments/paths.example.yaml experiments/paths.yaml
```

```yaml
data_root: /path/to/data          # directory containing harmonized_hru_data.zarr
log_root: /path/to/logs           # experiment outputs (checkpoints, Optuna databases, logs, predictions)
hru_file: /path/to/data/hrus/hrus.gpkg  # HRU polygons, used for plotting
```

Experiment configs derive their paths from it via `${data_root}`/`${log_root}` interpolation; scripts, plotting and notebooks read it through `project/utils/paths.py`. The remaining keys are only needed to rebuild the dataset from raw data or for the analysis notebooks. Keep all keys of the template in your copy, even unused ones with their placeholder values; the configuration accepts exactly these keys.

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
| `routed` | DROP | Pretrained runoff model coupled with learned routing along the river network. |
| `instant` | DROP | As `routed`, but with instantaneous routing (area-weighted summation), as an ablation. |

`lumped`, `routed` and `instant` are run for two data splits: `hard` holds out a fixed set of catchments for testing, `random` splits at random (see `project/data/splitting.py`). Configs for an additional `easy` split, with a different set of held-out catchments, are included but not used in the paper.

### Running

Each experiment runs hyperparameter tuning with `tune.py` first. `xval.py` then saves the predictions of the best tuning checkpoint: despite its name, it does not retrain the model, and all experiments use a single fold. `prevah` is only tuned: `routed` and `instant` load its best model from `<log_root>/prevah` as their pretrained runoff model, so it has to be tuned before them.

```bash
# 1. Pretrain the runoff model (no splits).
pixi run python tune.py -c experiments/paths.yaml -c experiments/base_config.yaml -c experiments/prevah/config.yaml

# 2. Any other experiment, for <model> in lumped, routed, instant and <split> in random, hard.
pixi run python tune.py -c experiments/paths.yaml -c experiments/base_config.yaml \
    -c experiments/<model>/config.yaml -c experiments/<model>/<split>/config.yaml
pixi run python xval.py -c experiments/paths.yaml -c experiments/base_config.yaml \
    -c experiments/<model>/config.yaml -c experiments/<model>/<split>/config.yaml
```

Adding `-c experiments/fast_dev_run_config.yaml` runs a tiny version (2 trials, 1 epoch, short sequences) to check the setup. It writes to `<log_root>/fast_dev_run/`, so it never overwrites the results of real runs.

The analysis notebook `analysis/prevah_vs_runoff.ipynb` (comparison with PREVAH) additionally needs the `xval.py` run of `prevah`, which the other experiments do not:

```bash
pixi run python xval.py -c experiments/paths.yaml -c experiments/base_config.yaml -c experiments/prevah/config.yaml
```

On a SLURM cluster, `bash run_all.sh` submits all experiments and splits in the right order, with several tuning workers per experiment (see [SLURM](#slurm) for the one-time setup). To sanity-check your setup with a single fast, low-resource job first:

```bash
bash slurm/example_minimal.sh
```

> [!WARNING]
> `run_all.sh` and `experiments/*/run.sh` use `--tuning_engine.overwrite=True`, which deletes the existing results of these experiments under `<log_root>`.

### Results and figures

After the runs, two post-processing steps prepare the inputs of the notebooks:

- `bash slurm/combine_predictions.sh` combines the `xval.py` predictions of all experiments except `prevah` into `<log_root>/predictions_combined.zarr`. Set `predictions_file` in `experiments/paths.yaml` to this file.
- `bash slurm/compute_source_contribution.sh` computes the source contributions of the Aare outlet during the August 2007 flood, used by `analysis/case_study.ipynb`.

The notebooks in `analysis/` then produce the figures in `analysis/figures/`. The hillshades used as map backgrounds are included in `analysis/data/`; they are derived from MERIT Hydro ([Yamazaki et al., 2019](https://doi.org/10.1029/2019WR024873)) and licensed under the [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/). Start Jupyter in the pixi environment with:

```bash
pixi run jupyter lab
```

> [!NOTE]
> A mapping from the notebooks to the figures of the paper will be added once the paper is published.

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
- **Configuration:** every experiment config resolves through the CLI, test runs with `fast_dev_run_config.yaml` stay out of the real results, and no machine-specific paths are tracked outside `experiments/paths.yaml`.
- **Checkpoints:** the best tuning checkpoint is found, also after moving a run directory.

Two tests use the real dataset at `data_root` and are skipped automatically if it isn't found. The test suite, ruff and a check that notebooks are committed without outputs run on every push to `main` and on every pull request (see the badge above); there, the tests that need the dataset or a local `experiments/paths.yaml` are skipped.

## Citation

> [!NOTE]
> Citation information will be added here once the paper is published.

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE). The hillshades in `analysis/data/` are an exception: as data derived from MERIT Hydro, they are licensed under the [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
