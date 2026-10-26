# Author: Basil Kraft
# Created: 2025-07-16
# Description: CLI interface for configuring and running hyperparameter optimization and cross-validation
#              workflows using Optuna and PyTorch Lightning.

"""Custom LightningCLI wrapper for configuring and executing model training workflows with Optuna.

This module defines a specialized CLI class (`CLI`) that extends PyTorch Lightning’s `LightningCLI`.
It provides structured argument parsing, dynamic class instantiation (including Optuna samplers/pruners),
and integration of forced callbacks like early stopping and checkpointing. It is specifically designed
to support hyperparameter tuning and cross-validation via a `TuningEngine`.

Main Features:
    - Argument linking between data module and tuning engine (e.g., number of folds).
    - Dynamic instantiation of Optuna components (e.g., sampler, pruner) based on configuration.
    - Registration of early stopping and checkpointing callbacks with sensible defaults.
    - Compatibility with YAML or command-line configuration using `jsonargparse`.

Typical usage:
    cli = CLI()  # Instantiated and used inside a main() function
    cli.tuning_engine.setup(which="all")  # Prepare directories and optuna database.
    cli.tuning_engine.tune()  # Run hyperparameter tuning.
    cli.tuning_engine.xval()  # Run cross-validation.
"""

import importlib
import warnings

import lightning.pytorch as pl
import optuna
from jsonargparse import namespace_to_dict
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

from project.utils.paths import PATH_KEYS

from .utils import PredictionWriter

# Ignore anticipated PL warnings.
warnings.filterwarnings("ignore", ".*infer the indices fetched for your dataloader.*")
warnings.filterwarnings("ignore", ".*You requested to overfit.*")


class CLI(LightningCLI):
    """A custom LightningCLI for configuring and executing model training using Optuna and PyTorch Lightning.

    This class wraps around the PyTorch LightningCLI to provide additional argument parsing,
    callback configuration, and dynamic instantiation of tuning-related components such as
    samplers and pruners.

    Key Features:
        - Adds and links arguments for tuning configuration.
        - Sets up early stopping and model checkpointing callbacks by default.
        - Automatically constructs the TuningEngine using the parsed configuration.
        - Integrates seamlessly with Optuna for tuning and cross-validation experiments.
    """

    def __init__(self, **kwargs):
        """Initialize the class with the provided keyword arguments.

        This constructor calls the superclass initializer with specific arguments to set up
        the LightningModule and LightningDataModule, along with additional configuration options.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the superclass initializer.
        """

        kwargs.setdefault("parser_kwargs", {}).setdefault("parser_mode", "omegaconf+")

        super().__init__(
            pl.LightningModule,
            pl.LightningDataModule,
            run=False,
            subclass_mode_model=True,
            subclass_mode_data=True,
            **kwargs,
        )

    def add_arguments_to_parser(self, parser):
        """Add custom arguments and argument blocks to the provided parser for configuration.

        This method first adds base arguments from the superclass, then registers arguments for the
        TuningEngine class under the "tuning_engine" namespace. It links relevant arguments (such as
        number of folds and monitor metrics) between components. It also adds a search space argument
        for hyperparameter tuning, and registers forced callbacks for early stopping and model checkpointing,
        setting their default values and linking their monitored metrics to the tuning engine.

        Args:
            parser: The argument parser object to which arguments will be added.
        """

        # Add LightningCLI args first
        super().add_arguments_to_parser(parser)

        # Centralized paths from experiments/paths.yaml. Experiment configs reference them via `${data_root}`/
        # `${log_root}` interpolation. All keys are registered, including ones the CLI does not use itself, so that
        # experiments/paths.yaml can be passed as a `-c` config without jsonargparse rejecting unknown keys.
        for key, description in PATH_KEYS.items():
            parser.add_argument(f"--{key}", type=str, help=description)

        # Avoid circular import
        from project.core.tuning_engine import TuningEngine

        # Register the tuning_engine argument block
        parser.add_class_arguments(TuningEngine, "tuning_engine")

        parser.link_arguments(source="data.init_args.num_folds", target="tuning_engine.num_folds")

        # Forced callbacks
        parser.add_lightning_class_args(EarlyStopping, "early_stopping_callback")
        parser.link_arguments(source="tuning_engine.monitor", target="early_stopping_callback.monitor")
        parser.set_defaults(
            {
                "early_stopping_callback.patience": 40,
                "early_stopping_callback.mode": "min",
                "early_stopping_callback.min_delta": 0.0,
                "early_stopping_callback.check_finite": True,
            }
        )
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint_callback")
        parser.link_arguments(source="tuning_engine.monitor", target="checkpoint_callback.monitor")
        parser.set_defaults(
            {
                "checkpoint_callback.mode": "min",
                "checkpoint_callback.save_last": True,
                "checkpoint_callback.save_top_k": 1,
                "checkpoint_callback.filename": "best",
            }
        )
        parser.add_lightning_class_args(PredictionWriter, "prediction_writer")
        parser.set_defaults(
            {
                "prediction_writer.write_all_variables": False,
            }
        )

        # ---------- Custom configuration ----------
        # Link input dimensionality
        parser.link_arguments("data.num_stat_features", "model.init_args.num_stat_features", apply_on="instantiate")
        parser.link_arguments(
            "data.num_stat_routing_features", "model.init_args.num_stat_routing_features", apply_on="instantiate"
        )
        parser.link_arguments("data.num_dyn_features", "model.init_args.num_dynamic_features", apply_on="instantiate")
        parser.link_arguments("data.prediction_length", "model.init_args.prediction_length", apply_on="instantiate")

        # Link normalization arguments
        parser.link_arguments(
            "data.norm_args_stat_features", "model.init_args.norm_args_stat_features", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.norm_args_stat_routing_features",
            "model.init_args.norm_args_stat_routing_features",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.norm_args_dyn_features", "model.init_args.norm_args_dyn_features", apply_on="instantiate"
        )
        parser.link_arguments("data.norm_args_targets", "model.init_args.norm_args_targets", apply_on="instantiate")
        parser.link_arguments("tuning_engine.pruner.init_args.max_resource", "model.init_args.lr_cycle_epochs")

    def before_instantiate_classes(self):
        """Prepare and instantiates the tuning engine and its dependencies before class instantiation.

        This method performs the following steps:
        1. Imports the TuningEngine class to avoid circular imports.
        2. Retrieves and converts the tuning engine configuration from the main config.
        3. Extracts and removes the 'sampler' and 'pruner' configurations from the tuning engine config.
        4. Instantiates the sampler and pruner objects using the provided configurations.
        5. Instantiates the TuningEngine with the created sampler, pruner, and remaining configuration parameters,
           and assigns it to `self.tuning_engine`.

        Raises:
            Any exceptions raised by `self.class_instantiate` or TuningEngine initialization.
        """

        # Currently no-op
        super().before_instantiate_classes()

        # Avoid circular import
        from project.core.tuning_engine import TuningEngine

        tuner_cfg_raw = self.config["tuning_engine"]

        # Convert everything to a real dict
        tuner_cfg = namespace_to_dict(tuner_cfg_raw)

        # Extract nested configs
        sampler_cfg = tuner_cfg.pop("sampler", None)
        pruner_cfg = tuner_cfg.pop("pruner", None)

        sampler: optuna.samplers.BaseSampler = self.class_instantiate(sampler_cfg)
        pruner: optuna.pruners.BasePruner = self.class_instantiate(pruner_cfg)

        self.tuning_engine = TuningEngine(sampler=sampler, pruner=pruner, **tuner_cfg)

    def class_instantiate(self, cfg):
        """Dynamically instantiate a class from a configuration dictionary.

        Args:
            cfg (dict): Configuration dictionary with the following keys:
                - 'class_path' (str): The full import path to the class (e.g., 'module.submodule.ClassName').
                - 'init_args' (dict, optional): Dictionary of keyword arguments to pass to the class constructor.

        Returns:
            object: An instance of the specified class, initialized with the provided arguments.

        Raises:
            ValueError: If cfg is not a dictionary.
        """

        if not isinstance(cfg, dict):
            raise ValueError("Expected dict with keys 'class_path' and optional 'init_args'.")

        class_path = cfg["class_path"]
        init_args = cfg.get("init_args", {})

        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**init_args)
