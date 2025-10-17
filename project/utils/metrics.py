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


def _relscale(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: Dims) -> xr.DataArray:

    relscale = (mod.std(dim=dim, skipna=True) - obs.std(dim=dim, skipna=True)) / obs.std(dim=dim, skipna=True) * 100

    relscale.attrs['units'] = '%'

    return relscale.compute()


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


def _get_xflow_bias(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: str,
        q_low: float,
        q_up: float) -> xr.DataArray:

    if not (0 <= q_low <= 1):
        raise ValueError(
            f'`q_low` must be in range [0, 1], is {q_low}.'
        )
    if not (0 <= q_up <= 1):
        raise ValueError(
            f'`q_up` must be in range [0, 1], is {q_up}.'
        )
    if q_up <= q_low:
        raise ValueError(
            f'`q_up` must be larger than `q_low`, is {q_up} <= {q_low}.'
        )

    only_mod_dims = tuple(np.setdiff1d(np.array(mod.dims), np.array(obs.dims)))
    n_only_mod_dims = len(only_mod_dims)

    obs_da = obs.transpose(..., dim)
    mod_da = mod.transpose(*only_mod_dims, ..., dim)

    da_res = mod_da.isel({dim: 0}).copy()

    for idx in np.ndindex(mod_da.shape[:-1]):

        mod_i = mod_da[idx].values

        obs_i = obs_da[idx[n_only_mod_dims:]].values

        sort_idx = np.argsort(obs_i)

        n = np.isfinite(obs_i).sum()
        low_i = int(np.floor(q_low * n))
        up_i = int(np.floor(q_up * n))
        sort_idx_in_range = sort_idx[low_i:up_i]

        obs_in_range = obs_i[sort_idx_in_range]
        mod_in_range = mod_i[sort_idx_in_range]

        perc_bias = 100 * (mod_in_range - obs_in_range).sum() / obs_in_range.sum()

        da_res[idx] = perc_bias

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

    flv = _get_xflow_bias(obs=obs, mod=mod, dim=dim, q_low=0.0, q_up=0.05).compute()

    return flv


def _fhv(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: str) -> xr.DataArray:

    if not isinstance(dim, str):
        raise ValueError(
            '`fhv` not implemented for more than on dimension.'
        )

    fhv = _get_xflow_bias(obs=obs, mod=mod, dim=dim, q_low=0.98, q_up=1.00).compute()

    return fhv


METRIC_MAPPING: dict[str, dict] = dict(
    bias={'func': _bias, 'name': 'Bias', 'direction': 'none'},
    absbias={'func': _absbias, 'name': 'Absolute bias', 'direction': 'min'},
    relbias={'func': _relbias, 'name': '%Bias', 'direction': 'none'},
    relscale={'func': _relscale, 'name': '%Scale Bias', 'direction': 'none'},
    mse={'func': _mse, 'name': 'Mean squared error (MSE)', 'direction': 'min'},
    bias2={'func': _bias2, 'name': 'Squared bias', 'direction': 'min'},
    varerr={'func': _varerr, 'name': 'Variance error', 'direction': 'min'},
    phaseerr={'func': _phaseerr, 'name': 'Phase error', 'direction': 'min'},
    rmse={'func': _rmse, 'name': 'Root mean squared error (RMSE)', 'direction': 'min'},
    nse={'func': _nse, 'name': 'Modeling efficiency (NSE)', 'direction': 'max'},
    nnse={'func': _nnse, 'name': 'Normalized modeling efficiency (NNSE)', 'direction': 'max'},
    kge={'func': _kge, 'name': 'Kling-Gupta efficiency (KGE)', 'direction': 'max'},
    r={'func': _r, 'name': 'Pearson\'s correlation (r)', 'direction': 'max'},
    flv={'func': _flv, 'name': '%Bias$_\mathrm{lowflow}$ (<Q.5)', 'direction': 'none'},
    fhv={'func': _fhv, 'name': '%Bias$_\mathrm{highflow}$ (>Q.98)', 'direction': 'none'},
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
