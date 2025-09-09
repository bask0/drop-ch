"""Implements common metrics for model evaluation."""

import numpy as np
import xarray as xr
from xarray.core.types import Dims


def common_mask(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: Dims = None,
        drop_empty: bool = True) -> tuple[xr.DataArray, xr.DataArray]:
    """Apply common mask to two datasets, such that they have NaN values at the same locations.

    Args:
        obs (xr.DataArray): Observed data.
        mod (xr.DataArray): Modeled data.
        dim (Dims): Dimension(s) to apply the mask on. Default is None.
        drop_empty (bool): If True, drop dimensions where all values are NaN after masking
    """

    only_mod_dims = tuple(np.setdiff1d(np.array(mod.dims), np.array(obs.dims)))
    mask = mod.notnull().any(only_mod_dims) & obs.notnull()

    if drop_empty:
        all_miss = mask.any(dim).compute()
        mask = mask.where(all_miss, drop=True)

    obs = obs.where(mask).compute()
    mod = mod.where(mask).compute()

    return obs, mod


def _bias(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: Dims) -> xr.DataArray:

    bias = mod.mean(dim=dim, skipna=True) - obs.mean(dim=dim, skipna=True)

    return bias.compute()


def _relbias(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: Dims) -> xr.DataArray:

    relbias = (mod.mean(dim=dim, skipna=True) - obs.mean(dim=dim, skipna=True)) / obs.mean(dim=dim, skipna=True) * 100

    relbias.attrs['units'] = '%'

    return relbias.compute()


def _absbias(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: Dims) -> xr.DataArray:

    absbias: xr.DataArray = np.abs(_bias(obs=obs, mod=mod, dim=dim))  # type: ignore

    return absbias.compute()


def _mse(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: Dims) -> xr.DataArray:

    return ((mod - obs)**2).mean(dim=dim).compute()


def _bias2(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: Dims) -> xr.DataArray:

    return _bias(obs=obs, mod=mod, dim=dim) ** 2


def _varerr(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: Dims) -> xr.DataArray:

    return (mod.std(dim=dim) - obs.std(dim=dim)) ** 2


def _phaseerr(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: Dims) -> xr.DataArray:

    return (1.0 - _r(obs=obs, mod=mod, dim=dim)) * 2.0 * mod.std(dim=dim) * obs.std(dim=dim)


def _rmse(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: Dims) -> xr.DataArray:

    return (_mse(obs=obs, mod=mod, dim=dim) ** 0.5).compute()


def _r(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: Dims) -> xr.DataArray:

    res = xr.corr(obs, mod, dim=dim).compute()

    res.attrs['units'] = '%'

    return res


def _nse(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: Dims) -> xr.DataArray:

    top = ((mod - obs)**2).sum(dim=dim, skipna=True)
    bottom = ((obs - obs.mean(dim=dim, skipna=True))**2).sum(dim=dim, skipna=True)

    nse = 1 - top / bottom

    nse = nse.where(mod.notnull().any(dim), np.nan)

    nse.attrs['units'] = '-'

    return nse.compute()


def _nnse(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: Dims) -> xr.DataArray:

    nse = _nse(obs=obs, mod=mod, dim=dim)

    nse.attrs['units'] = '-'

    return (1 / (2 - nse)).compute()


def _kge(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: Dims) -> xr.DataArray:

    r = _r(obs, mod, dim=dim)
    alpha = mod.std(dim=dim) / obs.std(dim=dim)
    beta = mod.mean(dim=dim) / obs.mean(dim=dim)

    value = ((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    value.attrs['units'] = '-'

    return (1 - value ** 0.5).compute()


def _get_sorted_fraction(x, fraction, direction):
    """Get a fraction of the smallest or largest values along the last axis.

    Note: Instead of cutting the fraction, we fill in NaNs to dead with different number of values.

    Args:
        x: a numpy array
        fraction: the fraction of lowest (if `direction='low'`) or highest (if `direction='high'`) values to extract.
        direction: 'low' for lowest or 'high' for largest values.

    Returns:
        A numpy Array with lowest or largest values up to given fraction, filled with NaN where above threshold.
    """

    if not (0 <= fraction <= 1):
        raise ValueError(
            f'fraction must be in range [0, 1], is {fraction}.'
        )

    for idx in np.ndindex(x.shape[:-1]):

        if direction == 'high':
            x_sorted = -np.sort(-x[idx])
        elif direction == 'low':
            x_sorted = np.sort(x[idx])
        else:
            raise ValueError(
                f'direcion mus be \'low\' or \'high\', is \'{direction}\'.'
            )

        num_valid = np.isfinite(x_sorted).sum()
        num_cut = np.round(fraction * num_valid).astype(int)
        x_sorted[num_cut:] = np.nan

        x[idx] = x_sorted

    return x


def _get_xflow_bias(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: str,
        fraction: float,
        direction: str) -> xr.DataArray:

    obs = obs.transpose(..., dim)
    mod = mod.transpose(..., dim)

    obs_da, mod_da = common_mask(obs=obs, mod=mod, dim=dim)

    obs_s = _get_sorted_fraction(x=obs_da.values, fraction=fraction, direction=direction)
    mod_s = _get_sorted_fraction(x=mod_da.values, fraction=fraction, direction=direction)

    obs_mean = np.nanmean(obs_s, axis=-1)
    mod_mean = np.nanmean(mod_s, axis=-1)

    res = (mod_mean - obs_mean) / (obs_mean + 1e-6)

    da_res = obs_da.isel({dim: 0}).drop_vars(dim).copy().load()
    da_res.values[:] = res * 100

    da_res.attrs['units'] = '%'

    return da_res


def _flv(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: str) -> xr.DataArray:

    if not isinstance(dim, str):
        raise ValueError(
            '`flv` not implemented for more than on dimension.'
        )

    flv = _get_xflow_bias(obs=obs, mod=mod, dim=dim, fraction=0.05, direction='low').compute()
    # flv = flv.where(mod.notnull().any(dim), np.nan)

    return flv


def _fhv(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: str) -> xr.DataArray:

    if not isinstance(dim, str):
        raise ValueError(
            '`fhv` not implemented for more than on dimension.'
        )

    fhv = _get_xflow_bias(obs=obs, mod=mod, dim=dim, fraction=0.02, direction='high').compute()
    # fhv = fhv.where(mod.notnull().any(dim), np.nan)

    return fhv


METRIC_MAPPING: dict[str, dict] = dict(
    bias={'func': _bias, 'name': 'Bias', 'direction': 'none'},
    absbias={'func': _absbias, 'name': 'Absolute bias', 'direction': 'min'},
    relbias={'func': _relbias, 'name': '%Bias', 'direction': 'none'},
    mse={'func': _mse, 'name': 'Mean squared error', 'direction': 'min'},
    bias2={'func': _bias2, 'name': 'Squared bias', 'direction': 'min'},
    varerr={'func': _varerr, 'name': 'Variance error', 'direction': 'min'},
    phaseerr={'func': _phaseerr, 'name': 'Phase error', 'direction': 'min'},
    rmse={'func': _rmse, 'name': 'Root mean squared error', 'direction': 'min'},
    nse={'func': _nse, 'name': 'Modeling efficiency', 'direction': 'max'},
    nnse={'func': _nnse, 'name': 'Normalized modeling efficiency', 'direction': 'max'},
    kge={'func': _kge, 'name': 'Kling-Gupta efficiency', 'direction': 'max'},
    r={'func': _r, 'name': 'Pearson\'s correlation', 'direction': 'max'},
    flv={'func': _flv, 'name': '%Bias FLV', 'direction': 'none'},
    fhv={'func': _fhv, 'name': '%Bias FHV', 'direction': 'none'},
)


def compute_metrics(
        obs: xr.DataArray,
        mod: xr.DataArray,
        metrics: str | list[str] = 'all',
        dim: Dims = None,
        drop_empty: bool = True) -> xr.Dataset:
    """Compute metrics for model evaluation.

    Args:
        obs (xr.DataArray): Observed data.
        mod (xr.DataArray): Modeled data.
        metrics (str | list[str]): Metrics to compute. If 'all', computes all available metrics. Defaults to 'all'.
        dim (Dims): Dimension(s) to apply the metrics on. Default is None,
        drop_empty (bool): If True, drop dimensions where all values are NaN after masking. Defaults to True.

    Returns:
        xr.Dataset: Dataset containing computed metrics.

    """

    if metrics == 'all':
        metrics = list(METRIC_MAPPING.keys())

    if dim is None:
        dim = [str(dim) for dim in mod.dims if dim in obs.dims]

    metrics = [metrics] if isinstance(metrics, str) else metrics

    missing_metrics = []
    for metric in metrics:
        if metric not in METRIC_MAPPING:
            missing_metrics.append(metric)

    if len(missing_metrics) > 0:
        raise ValueError(
            f'the following metrics are not implemented: `{"`, `".join(missing_metrics)}`'
        )

    obs_da, mod_da = common_mask(obs=obs, mod=mod, dim=dim, drop_empty=drop_empty)

    ds = xr.Dataset()

    for metric in metrics:
        func = METRIC_MAPPING[metric]['func']
        name = METRIC_MAPPING[metric]['name']
        direction = METRIC_MAPPING[metric]['direction']

        da = func(obs=obs_da, mod=mod_da, dim=dim)

        da.attrs['long_name'] = name
        da.attrs['direction'] = direction

        ds[metric] = da

    return ds
