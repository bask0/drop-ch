"""Implementation of utility functions for plotting."""

import os
import shutil
import warnings
from glob import glob
from os import PathLike
from pathlib import Path
from typing import Literal

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import optuna
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Patch
from pyproj import CRS

from project.utils.data import load_xval_test_set
from project.utils.metrics import compute_metrics
from project.utils.paths import load_paths_config

# TODO: Make this more flexible, but tuning_engine does not have access to data paths.
_hru_file = load_paths_config().get("hru_file")
HRU_FILE_CANDIDATES = [Path(_hru_file)] if _hru_file else []


def despine(ax: Axes, which: str = "L", remove_ticks: bool = False) -> None:
    """Remove spines from the plot.

    Args:
        ax (Axes): The axes to modify.
        which (str): Which spines to remove. Options are 'all' for all spines, or 'L' for left and bottom spines.
            Defaults to 'all'.
        remove_ticks (bool): If True, also remove ticks and labels from the axes.

    Raises:
        NotImplementedError: If `which` is not 'all' or 'L'.

    """

    if which == "all":
        spines = ["right", "top", "bottom", "left"]
    elif which == "L":
        spines = ["top", "right"]
    else:
        raise NotImplementedError(f"`which`={which} not implemented.")

    ax.spines[spines].set_visible(False)

    if remove_ticks:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def cbar_halo(
    cbar: Colorbar | None = None,
    ax: Axes | None = None,
    which: Literal["x", "y"] | None = None,
    linewidth: float = 5,
    color: str = "w",
    labelsize: int = 8,
    ticksize: int = 6,
    flip_label: bool = True,
    **kwargs,
) -> None:
    """Add a halo effect to the colorbar axis labels and tick labels.

    Args:
        cbar (Colorbar): The colorbar to modify. If None, `ax` must be provided.
        ax (Axes | None): The axes containing the colorbar. If `cbar` is None, this must be provided
            together with `which`.
        which (Literal['x', 'y']): Which axis to apply the halo effect to ('x' or 'y'). Only required
            if `cbar` is None.
        linewidth (float): The linewidth of the halo effect. Defaults to 5.
        color (str): The color of the halo effect. Defaults to 'white'.
        labelsize (int): Font size for the axis label. Defaults to 8.
        ticksize (int): Font size for the tick labels. Defaults to 6.
        flip_label (bool): If True and `which` is 'y', flip the label
        **kwargs: Additional keyword arguments to pass to the path effect.

    """

    halo = [path_effects.withStroke(linewidth=linewidth, foreground=color, **kwargs)]

    if cbar is None:
        if ax is None or which is None:
            raise ValueError("If `cbar` is None, both `ax` and `which` must be provided.")
    else:
        ax = cbar.ax

        if cbar.orientation == "vertical":
            which = "y"
        else:
            which = "x"

    if which == "x":
        axis = ax.xaxis
        ticklabels = ax.get_xticklabels()
    elif which == "y":
        axis = ax.yaxis
        ticklabels = ax.get_yticklabels()
    else:
        raise ValueError(f'`which` must be "x" or "y", got {which}.')

    ax.tick_params(axis=which, labelsize=ticksize)

    axis.label.set_size(labelsize)

    if flip_label:
        axis.set_label_position("top" if which == "x" else "left")

    axis.label.set_path_effects(halo)

    for t in ticklabels:
        t.set_path_effects(halo)


OPTUNA_PLOTS_TUNING = [
    "plot_slice",
    "plot_contour",
    "plot_rank",
    "plot_intermediate_values",
    "plot_optimization_history",
    "plot_parallel_coordinate",
    "plot_param_importances",
    "plot_timeline",
]

OPTUNA_PLOTS_XVAL = [
    "plot_timeline",
]


def _optuna_plots(study: optuna.Study, out_dir: str | PathLike, is_xval: bool = False) -> None:
    if is_xval:
        optuna_plots = OPTUNA_PLOTS_XVAL
    else:
        optuna_plots = OPTUNA_PLOTS_TUNING

    for optuna_plot in optuna_plots:
        try:
            fig = getattr(optuna.visualization, optuna_plot)(study)

            fig.write_image(os.path.join(out_dir, optuna_plot + ".png"), scale=2, width=1600, height=1200)
        except RuntimeError as e:
            print("The following RuntimeError was raised (and ignored) for plot '{optuna_plot}':\n", e)
        except ValueError as e:
            print("The following ValueError: was raised (and ignored) for plot '{optuna_plot}':\n", e)


def plot_optuna_summary(study_path: str | PathLike, study_name: str | None = None, is_xval: bool = False) -> None:
    """Create and save optuna plots for a given study.

    Args:
        study_path (str | PathLike): Path to the study database.
        study_name (str | None): Name of the study. If None, checks whether the storage contains a single study,
            and if so loads that study. Defaults to None.
        is_xval (bool): If True, use cross-validation plots. Defaults to False.

    """
    base_dir = os.path.dirname(study_path)
    plot_dir = os.path.join(base_dir, "optuna_plots")
    os.makedirs(plot_dir, exist_ok=True)

    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_path}")

    _optuna_plots(study=study, out_dir=plot_dir, is_xval=is_xval)


def _get_cdf(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray, float]:
    da = da.load()
    da = da.where(da.notnull(), drop=True)
    bins = [*sorted(da.values), np.inf]
    count, bins = np.histogram(da, bins=bins)
    bins = bins[:-1]
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    xloc = da.median().compute()

    return bins, cdf, xloc.item()


def plot_cdf(
    ds: xr.Dataset,
    ours_name: str = "RunoffNet (ours)",
    save_path: str | PathLike | None = None,
    title_postfix: str = "",
    col: str = "#1E88E5",
) -> None:
    """Plot the empirical cumulative density function (CDF) of test set metrics.

    Args:
        ds (xr.Dataset): Dataset containing the metrics to plot.
        ours_name (str): Name to use in the plot legend for the model being evaluated.
            Defaults to 'RunoffNet (ours)'.
        save_path (str | PathLike | None): If provided, the plot will be saved to this path.
            If None, the plot will not be saved. Defaults to None.
        title_postfix (str): Postfix to add to the plot title. Defaults to ''.
        col (str): Color to use for the plot lines. Defaults to '#1E88E5'.

    """

    metrics = list(ds.data_vars)

    num_metrics = len(metrics)

    fig, axes = plt.subplots(1, num_metrics, figsize=(3 * num_metrics, 4), sharey="row", squeeze=False, gridspec_kw={})

    annot_kwargs = dict(textcoords="offset points", ha="center", va="bottom", color="0.2", fontsize=9)

    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        da = ds[metric]

        name = da.attrs.get("long_name", metric)

        bins, cdf, xloc = _get_cdf(da)
        ax.plot(bins, cdf, label=ours_name, color=col, alpha=0.7)
        ax.axvline(xloc, ymin=0, ymax=0.5, color=col, ls=":", alpha=0.7)
        ax.annotate(
            f"{xloc:0.2f}",
            xy=(xloc, 0.5),
            xytext=(-40, +15),
            bbox=dict(boxstyle="round,pad=0.2", fc=col, ec="none", alpha=0.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=col, alpha=0.7),
            **annot_kwargs,
        )

        ax.axhline(0.5, ls=":", color="k", alpha=0.7)

        ax.set_xlabel(name)
        despine(ax)

    axes[0, 0].set_ylabel("Empirical cumulative density")
    axes[0, -1].legend(frameon=False, fontsize=9)

    fig.suptitle("Station-level model comparison" + title_postfix)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def _infer_obs_pred_vars(ds: xr.Dataset, obs_var: str = "auto", pred_var: str = "auto") -> tuple[str, str]:
    """Infer observation and prediction variable names from the dataset attributes.

    Args:
        ds (xr.Dataset): Dataset containing the variables.
        obs_var (str): Observation variable name. If 'auto', it will be inferred from the dataset attributes.
            Defaults to 'auto'.
        pred_var (str): Prediction variable name. If 'auto', it will be inferred from the dataset attributes.
            Defaults to 'auto'.

    Returns:
        tuple[str, str]: Tuple containing the observation and prediction variable names.

    Raises:
        ValueError: If the dataset does not contain the required attributes or if the variables are not found.

    """

    if obs_var == "auto":
        if "obs_var" not in ds.attrs:
            raise ValueError(
                "Dataset does not contain attribute `obs_var`, cannot infer automatically. Pass `obs_var` explicitly."
            )
        obs_var = ds.attrs["obs_var"]

    if obs_var not in ds:
        raise ValueError(f"Observation variable `{obs_var}` not found in dataset.")

    if pred_var == "auto":
        if "pred_var" not in ds.attrs:
            raise ValueError(
                "Dataset does not contain attribute `pred_var`, cannot infer automatically. Pass `pred_var` explicitly."
            )
        pred_var = ds.attrs["pred_var"]

    if pred_var not in ds:
        raise ValueError(f"Observation variable `{pred_var}` not found in dataset.")

    return obs_var, pred_var


def plot_metric_cdf(
    xval_dir: str | PathLike, metrics: list[str] | None = None, obs_var: str = "auto", pred_var: str = "auto"
) -> None:
    """Plot the empirical cumulative density function (CDF) of metrics from cross-validation results.

    The save path is dervied from the `xval_dir` if not provided: [xval_dir]/'metric_plot.png'

    Args:
        xval_dir (str | PathLike): Directory containing the cross-validation results.
        metrics (list[str] | None): List of metrics to plot. If None, defaults to
            ['kge', 'r', 'relbias', 'flv', 'fhv'].
        obs_var (str): Observation variable name. If 'auto', it will be inferred from the dataset attributes.
            Defaults to 'auto'.
        pred_var (str): Prediction variable name. If 'auto', it will be inferred from the dataset attributes.
            Defaults to 'auto'.

    """

    xval_dir = Path(xval_dir)

    if metrics is None:
        metrics = ["kge", "kge_alpha", "kge_beta", "r", "flv", "fhv"]

    ds = load_xval_test_set(xval_dir)

    obs_var, pred_var = _infer_obs_pred_vars(ds, obs_var, pred_var)

    m = compute_metrics(obs=ds[obs_var], mod=ds[pred_var], metrics=metrics, dim="time")
    plot_cdf(ds=m, save_path=xval_dir / "metric_plot.png", title_postfix=" (cross-validation test set)")


def plot_predictions(
    xval_dir: str | PathLike, obs_var: str = "auto", pred_var: str = "auto", num_plot: int = 10
) -> None:
    """Plot observed and predicted time series for each HRU in the cross-validation test set.

    The save path is dervied from the `xval_dir` if not provided: [xval_dir]/ts_plots/[hru_id].png

    Args:
        xval_dir (str | PathLike): Directory containing the cross-validation results.
        obs_var (str): Observation variable name. If 'auto', it will be inferred from the dataset attributes.
            Defaults to 'auto'.
        pred_var (str): Prediction variable name. If 'auto', it will be inferred from the dataset attributes.
            Defaults to 'auto'.
        num_plot (int): Number of HRUs to plot. Defaults to 10.
    """

    xval_dir = Path(xval_dir)

    save_dir = xval_dir / "ts_plots"
    if save_dir.exists():
        shutil.rmtree(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    ds = load_xval_test_set(xval_dir)

    obs_var, pred_var = _infer_obs_pred_vars(ds, obs_var, pred_var)

    units = ds[obs_var].attrs.get("units", "")
    long_name = ds[obs_var].attrs.get("long_name", obs_var)

    col_obs = "k"
    col_mod = "#1E88E5"

    hrus = ds.hru.where(ds.has_obs.load(), drop=True)

    for i, hru in enumerate(hrus):
        if i == num_plot:
            break

        save_path = save_dir / f"{int(hru.item())}.png"

        ds_hru = ds.sel(hru=hru)

        fig, ax = plt.subplots(figsize=(6, 3))

        ds_hru[obs_var].plot(color=col_obs, lw=1.0, label="Observed", ax=ax)  # type: ignore
        ds_hru[pred_var].plot(color=col_mod, lw=1.0, label="Predicted", ax=ax)  # type: ignore

        ax.set_title(f"HRU #{int(hru.item())}")
        ax.set_ylabel(long_name + f" ({units})")
        despine(ax)
        ax.legend(frameon=False, loc="upper center", ncols=2)

        fig.tight_layout()

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close(fig)


def plot_xval_sets(xval_dir: str | PathLike) -> None:
    """Plot cross-validation sets on a map.

    Args:
        xval_dir (str | PathLike): Directory containing the cross-validation results.
    """

    color_mapping = {"None": "#FFFFFF", "Train": "#FFC107", "Valid": "#1E88E5", "Test": "#D81B60"}

    hru_path = None
    for candidate in HRU_FILE_CANDIDATES:
        if candidate.exists():
            hru_path = candidate
            break

    if hru_path is None:
        warnings.warn(
            "Could not find HRU file, `plot_xval_sets` not executed. The following paths are currently checked "
            f"(defined in {__file__}.HRU_FILE_CANDIDATES):\n"
            "\n".join(map(str, HRU_FILE_CANDIDATES)),
            stacklevel=2,
        )
        return

    poly = gpd.read_file(hru_path, use_arrow=True).set_index("EZGNR")

    paths = glob(os.path.join(xval_dir, "fold_*/preds.zarr"))

    for path in paths:
        ds = xr.open_zarr(path)
        poly["set"] = ""
        poly.loc[ds.hru, "set"] = [list(color_mapping.values())[v] for v in ds.subset.values]

        fig, ax = plt.subplots()

        poly.plot(ec="k", lw=0.1, legend=False, color=poly["set"], alpha=0.8, ax=ax)
        for _, s in enumerate([1, 2, 3]):
            hru_subset = ds.hru.where((ds.subset.load() == s) & ds.has_obs.load(), drop=True).values
            poly.loc[hru_subset].centroid.plot(
                ec="k", fc=list(color_mapping.values())[s], lw=0.9, markersize=10, alpha=0.8, ax=ax
            )

        despine(ax=ax, which="all", remove_ticks=True)

        # Create legend patches
        legend_patches = [Patch(color=color, label=category) for category, color in color_mapping.items()]

        # Add the legend to the plot
        ax.legend(handles=legend_patches, title="CV set", frameon=False, loc="upper left")

        save_path = Path(os.path.dirname(path)) / "cv_set.png"

        fig.tight_layout()

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close(fig)


class ProjectCustomExtent(ccrs.Projection):
    """Projections with custom (optionally beyond limits) extents."""

    def __init__(self, epsg, extent):
        """Initialize a custom projection with a specified extent."""

        self.xmin, self.xmax, self.ymin, self.ymax = extent
        super().__init__(CRS.from_epsg(epsg).to_string())

    @ccrs.Projection.boundary.getter
    def boundary(self):
        """Define the boundary of the projection based on the custom extent."""

        coords = (
            (self.x_limits[0], self.y_limits[0]),
            (self.x_limits[0], self.y_limits[1]),
            (self.x_limits[1], self.y_limits[1]),
            (self.x_limits[1], self.y_limits[0]),
        )
        return ccrs.sgeom.LineString(coords)

    @ccrs.Projection.x_limits.getter
    def x_limits(self):
        """Get the x-axis limits of the projection."""

        return self.xmin, self.xmax

    @ccrs.Projection.y_limits.getter
    def y_limits(self):
        """Get the y-axis limits of the projection."""

        return self.ymin, self.ymax
