<div align="center">

# Streamflow-Net

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
</div>

## Description

**Author:** Basil Kraft  
**Created:** July 17, 2025

Streamflow-Net is a data-driven neural network architecture for modeling and routing streamflow in distributed hydrological systems.

Designed to capture spatial and temporal dynamics of runoff generation and flow routing, Streamflow-Net leverages deep learning to simulate hydrological behavior across catchments with flexible, modular components.

## 🧠 Project Structure

```bash
# Root of your Git repository
.
├── project/                      # Installable package
│   ├── __init__.py
│   ├── core/                     # Core orchestration logic
│   │   ├── __init__.py
│   │   ├── cli_interface.py      # Custom LightningCLI wrapper
│   │   └── tuning_engine.py      # Core tuning and cross-validation logic
│   ├── models/                   # LightningModule subclasses
│   │   ├── __init__.py
│   │   └── ...                   # Add your models here
│   ├── data/                     # LightningDataModule subclasses
│   │   ├── __init__.py
│   │   └── ...                   # Add your data modules here
│   └── utils/                    # Optional utility functions
│       ├── __init__.py
│       └── ...
├── experiments/                  # Experiment definitions and outputs
│   ├── base_config.yaml          # Shared Hydra config (defaults)
│   ├── exp0/                     # One experiment instance
│   │   ├── config.yaml           # Hydra config for this experiment
│   │   └── logs/ → /log_dir/exp0 # Symlink to output logs of experiment 0
│   └── exp1/                     # Another experiment
│       ├── config.yaml
│       └── logs/ → /log_dir/exp1
├── tune.py                       # Entry point to CLI (Hydra or LightningCLI-based)
├── environment.yaml              # Conda + pip (editable) environment setup
├── pyproject.toml                # Package and build metadata (PEP 621)
└── README.md                     # Project overview and usage instructions

# External, system-level directory (not inside the repo)
/log_dir/exp0/
├── tune/                        # Hyperparameter tuning log dir
│   ├── trial_000/               # Trial 0 outputs
│   ├── trial_001/               # ...
│   └── optuna.db                # SQLite database for tuning
├── xval/                        # Cross-validation log dir
│   ├── fold_000/                # Fold 0 outputs
│   ├── fold_001/                # ...
│   └── optuna.db                # SQLite database for xval
/log_dir/exp1/
...
```

## How to run

First, install dependencies

```bash
# clone project   
git clone [URL]

# install project   
cd [REPO NAME]
conda env create -f environment.yaml
conda activate [ENV NAME]
```

You mey need to install chrome for plotly

`plotly_get_chrome`

Run hyperparameter tuning and cross validation.

```bash
python tune.py
```

### Citation

```text
@article{kraft_2025,
  title={Paper title},
  author={Basil Kraft},
  journal={Journal},
  year={2025}
}
```
