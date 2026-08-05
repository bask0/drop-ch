"""Compute gradients of streamflow at a given outlet with respect to runoff at all HRUs."""

from argparse import ArgumentParser
import logging
import math
from pathlib import Path
import numpy as np
import optuna
import torch
from tqdm import tqdm
import xarray as xr
import yaml
from zarr.codecs import BloscCodec, BloscShuffle

from project.data.datamodule import GraphDataModule
from project.nn.streamflow_net import StreamflowNet

logger = logging.getLogger('SourceContribution')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


argument_parser = ArgumentParser()
argument_parser.add_argument(
    '--optuna_tune_db',
    type=str,
    required=False,
    default='/cluster/work/climate/kraftb/runs/routed/random/tune/optuna.db',
    help='Path to the optuna tune database.'
)
argument_parser.add_argument(
    '--hru',
    type=int,
    required=False,
    default=145830,
    help='HRU ID of the outlet to compute gradients for. Default is 145830 (Aare).'
)
argument_parser.add_argument(
    '--time_window',
    type=int,
    required=False,
    default=100,
    help='Time window length (in timesteps) to compute gradients over. Default is 100.'
)
argument_parser.add_argument(
    '--time_warmup',
    type=int,
    required=False,
    default=365,
    help='Warmup time (in timesteps) before starting to compute gradients. Default is 500.'
)
argument_parser.add_argument(
    '--date_start',
    type=str,
    required=False,
    default='1993',
    help='Start date of the period to compute gradients for. Default and min allowed value is 1993.'
)
argument_parser.add_argument(
    '--date_end',
    type=str,
    required=False,
    default='2023',
    help='End date of the period to compute gradients for. Default and max allowed value is 2023.'
)
argument_parser.add_argument(
    '--out_dir',
    type=str,
    required=False,
    default='auto',
    help='Directory to save results to. If \'auto\', the path will be derived from the optuna_tune_db path, i.e., '
    '\'/some/path/to/tune/optuna.db\' -> \'/some/path/to/\'. The file path is then constructed as '
    '<out_dir>/source_contrib_<hru>.zarr. Default is \'auto\'.'
)


def main(args) -> None:
    """Compute contribution.

    Args:
        args: Command line arguments.

    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    optuna_path = Path(args.optuna_tune_db)
    log_dir = Path(optuna.load_study(
        study_name=None, storage=f'sqlite:///{optuna_path}').best_trial.user_attrs['log_dir'])
    checkpoint_path = optuna_path.parent / log_dir / 'checkpoints/best.ckpt'
    config_path = optuna_path.parent / log_dir / 'config.yaml'

    logger.info(f'Using checkpoint: {checkpoint_path}')

    snet = StreamflowNet.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    snet.eval()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']['init_args']

    logger.info(
        f'Using time range: {args.date_start} to {args.date_end} with window {args.time_window} and '
        f'warmup {args.time_warmup} for HRU {args.hru}.')

    data_config['batch_size'] = 1
    data_config['num_workers'] = 0
    data_config['ds_subset'] = {'time': slice(args.date_start, args.date_end)}

    datamodule = GraphDataModule(**data_config)

    dl = datamodule.predict_dataloader()

    hru_index = np.argwhere(dl.dataset.ds.hru.values == args.hru).item()  # type: ignore
    dl.dataset.sample_hrus = [hru_index]  # type: ignore
    batch = next(iter(dl))
    outlet_index = batch.mapping.item()

    time_window = args.time_window
    time_warmup = args.time_warmup

    n_time = batch.y.shape[1]
    n_windows = math.ceil((n_time - time_warmup) / time_window)

    if args.out_dir.lower() == 'auto':
        out_dir = optuna_path.parent
    else:
        out_dir = Path(args.out_dir)

    out_path = out_dir / f'source_contrib_{args.hru}.zarr'

    logger.info(f'Creating target dataset at \'{out_path}\'')

    ds_s = dl.dataset.ds.isel(hru=batch.subset)
    dummy_ds_with_grad_time = xr.full_like(ds_s.Q_obs_m3s, np.nan).expand_dims(
        lag=time_warmup + 1, axis=-1).assign_coords(
        lag=np.arange(time_warmup, -1, -1)).transpose('hru', 'time', 'lag')
    dummy_ds_no_grad_time = xr.full_like(ds_s.Q_obs_m3s, np.nan).transpose('hru', 'time')

    ds_dummy = ds_s[['area']].copy()

    ds_dummy['Q_pred_m3s'] = dummy_ds_no_grad_time
    ds_dummy['q_pred_m3s'] = dummy_ds_no_grad_time
    ds_dummy['q_grad'] = dummy_ds_with_grad_time
    ds_dummy['q_source_m3s'] = dummy_ds_with_grad_time

    encoding = {}

    for var in ds_dummy.data_vars:
        if 'lag' in ds_dummy[var].dims:
            chunks = (1000, 100, len(ds_dummy.lag))
        else:
            chunks = (1000, 100)

        encoding[var] = {
            'chunks': chunks,
            'compressors': BloscCodec(
                cname='zstd',
                clevel=3,
                shuffle=BloscShuffle.shuffle
            ),
        }

    ds_dummy.to_zarr(out_path, mode='w', encoding=encoding)

    for i in tqdm(range(n_windows), ncols=100, desc='Computing gradients'):

        # Window slice from windo start to end (including warmup)
        start = i * time_window
        end = min(start + time_warmup + time_window, n_time)

        # Subset batch to this window
        batch_device = batch.clone()
        batch_device.d = batch.d[:, start:end]
        batch_device.y = batch.y[:, start:end]
        batch_device.prec = batch.prec[:, start:end]
        batch_device.tair = batch.tair[:, start:end]
        batch_device.to(device)

        # Run model with full gradient graph (prec -> runoff_net -> q -> routing_net -> Q).
        q = snet.runoff_net.chunked_forward(batch=batch_device, scale_output=True, subset=None)
        q = snet.convert_q_to_m3s(q, batch_device)
        e = snet.evaporation_net(T=batch_device.tair, area=batch_device.hru_area,
                                 water_freq=batch_device.water_freq, subset=None)
        Q, *_ = snet.routing_net.route(batch=batch_device, q=q, e=e, subset=None)

        if i == 0:
            ds_dummy_region = ds_dummy.isel(time=slice(0, time_warmup))
            ds_dummy_region['Q_pred_m3s'].values = Q[:, :time_warmup, 0].detach().cpu()
            ds_dummy_region['q_pred_m3s'].values = q[:, :time_warmup, 0].detach().cpu()
            ds_dummy_region.drop_vars(['hru', 'hru_i', 'lag', 'area', 'q_grad', 'q_source_m3s']).to_zarr(
                out_path, region={'time': slice(0, time_warmup)})

        # Relative (to block) time steps to iterate over.
        times = range(time_warmup, min(time_warmup + time_window, Q.shape[1]))
        # Absolute (to ds_dummy) time steps to iterate over.
        abs_times = slice(i * time_window + time_warmup, (i + 1) * time_window + time_warmup)

        grad_per_timestep = []
        grad_times_q_per_timestep = []

        # Loop over timesteps, retaining the graph only on the first step
        for t, time in enumerate(times):
            # grad_output = dL/dQ = mask (because L = (Q*mask).sum())
            dq, = torch.autograd.grad(
                outputs=Q[outlet_index, time, :].sum(),
                inputs=q,                     # we only need the gradient w.r.t. q
                retain_graph=True
            )

            grads = dq.detach().cpu()
            grads_times_q = grads * q.detach().cpu()
            t0 = t
            t1 = t + time_warmup + 1
            grad_per_timestep.append(grads[:, t0:t1, :])
            grad_times_q_per_timestep.append(grads_times_q[:, t:t1, :])

        grad_per_timestep = torch.cat(grad_per_timestep, dim=-1).transpose(1, 2)
        grad_times_q_per_timestep = torch.cat(grad_times_q_per_timestep, dim=-1).transpose(1, 2)

        ds_dummy_region = ds_dummy.isel(time=abs_times)

        ds_dummy_region['Q_pred_m3s'].values = Q[:, time_warmup:, 0].detach().cpu()
        ds_dummy_region['q_pred_m3s'].values = q[:, time_warmup:, 0].detach().cpu()
        ds_dummy_region['q_grad'].values = grad_per_timestep
        ds_dummy_region['q_source_m3s'].values = grad_times_q_per_timestep

        # Write region.
        ds_dummy_region.drop_vars(['hru', 'hru_i', 'lag', 'area']).to_zarr(
            out_path, region={'time': abs_times}, mode='a')

        del batch_device, Q, q, e, dq   # clear refs
        torch.cuda.empty_cache()  # optional: reclaims cached blocks

    logger.info(f'Done. Find results at:\n\t\'{out_path}\'')


if __name__ == '__main__':
    args = argument_parser.parse_args()
    main(args)
