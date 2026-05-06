"""Evaluate the pretrained RunoffNet as a component of StreamflowNet (before fine-tuning).

Instantiates StreamflowNet with:
  - pretrained RunoffNet weights (from the lumped experiment)
  - randomly initialised RoutingModel (untrained)
  - instant routing (area-weighted aggregation, no learned transport)

Instant routing is used deliberately so that the untrained RoutingModel weights do not
contaminate the signal.  Any bias or error in the resulting Q comes from the RunoffNet.

Driven by the routed experiment config YAML (--streamflow_config), which provides the
feature lists, data path, and lumped experiment dir.

Usage
-----
python -m project.scripts.eval_pretrained_runoff_net \\
    --streamflow_config experiments/routed/random/config.yaml \\
    --output_path /tmp/pretrained_runoff_eval.nc
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml
from tqdm import tqdm

from project.data.datamodule import GraphDataModule
from project.nn.streamflow_net import StreamflowNet

# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--streamflow_config",
        type=str,
        default="experiments/routed/random/config.yaml",
        help="Routed experiment config YAML.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="pretrained_runoff_eval.nc",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_merged_config(config_path: str | Path) -> dict:
    """Load a config YAML and merge with its parent configs bottom-up.

    Walks up the directory tree collecting config.yaml files until it reaches
    the 'experiments' root, then merges them from outermost to innermost so
    that the most specific file wins.
    """
    config_path = Path(config_path).resolve()
    configs: list[dict] = []

    path = config_path
    while True:
        configs.append(load_yaml(path))
        parent_cfg = path.parent.parent / "config.yaml"
        if not parent_cfg.exists() or parent_cfg == path:
            break
        path = parent_cfg

    # Merge from outermost (least specific) to innermost (most specific)
    merged: dict = {}
    for cfg in reversed(configs):
        merged = deep_merge(merged, cfg)
    return merged


def data_kwargs_from_config(cfg: dict) -> dict:
    """Extract GraphDataModule init_args from a config dict, dropping unsupported keys."""
    raw = cfg["data"]["init_args"]
    drop = {"num_folds", "fold_idx", "load_ds_to_memory"}
    return {k: v for k, v in raw.items() if k not in drop}


# ── Inference ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_inference(model: StreamflowNet, dataloader, device: str):
    """Run instant_forward and inference_forward on every batch and collect results.

    instant_forward : pretrained RunoffNet → area-weighted aggregation (no routing weights)
    inference_forward: pretrained RunoffNet → randomly initialised RoutingModel
    """
    records: list[dict] = []

    for batch in tqdm(dataloader, desc="Inference"):
        batch = batch.to(device)  # type: ignore

        # Q_instant: area-weighted sum of HRU runoff (no learned transport)
        _, Q_instant = model.instant_forward(batch)

        # Q_routed: output of the randomly initialised RoutingModel
        Q_routed, *_ = model.inference_forward(batch)

        Q_instant = Q_instant.squeeze(-1)  # [N, T]
        Q_routed = Q_routed.squeeze(-1)    # [N, T]

        # batch.y is Q_obs_m3s [N, T, 1]
        y = batch.y.squeeze(-1)  # [N, T]

        hru_idx = batch.hru_idx.cpu().numpy()
        # time_slice is [[start, end]] after collation; broadcast start index to all HRUs
        t_start = int(batch.time_slice[0][0])

        records.append(
            {
                "hru_idx": hru_idx,
                "time_start": np.full(len(hru_idx), t_start),
                "Q_instant_m3s": Q_instant.cpu().numpy(),  # instant routing (no routing weights)
                "Q_routed_m3s": Q_routed.cpu().numpy(),    # randomly initialised RoutingModel
                "Q_obs_m3s": y.cpu().numpy(),              # observed streamflow
            }
        )

    return records


# ── Metrics ───────────────────────────────────────────────────────────────────


def print_summary(pred: np.ndarray, obs: np.ndarray, label: str) -> None:
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs_m, pred_m = obs[mask], pred[mask]
    if len(obs_m) == 0:
        print(f"[{label}] No valid pairs found.")
        return

    bias_abs = float(np.mean(pred_m - obs_m))
    bias_rel = bias_abs / float(np.mean(obs_m)) * 100.0
    rmse = float(np.sqrt(np.mean((pred_m - obs_m) ** 2)))
    obs_var = float(np.sum((obs_m - obs_m.mean()) ** 2))
    nse = 1.0 - float(np.sum((pred_m - obs_m) ** 2)) / obs_var if obs_var > 0 else float("nan")

    print(f"\n── {label} ─────────────────────────────────────────────────────────")
    print(f"  N valid pairs   : {len(obs_m):,}")
    print(f"  Mean obs (m³/s) : {obs_m.mean():.4f}")
    print(f"  Mean pred (m³/s): {pred_m.mean():.4f}")
    print(f"  Absolute bias   : {bias_abs:+.4f} m³/s")
    print(f"  Relative bias   : {bias_rel:+.2f} %")
    print(f"  RMSE            : {rmse:.4f} m³/s")
    print(f"  NSE             : {nse:.4f}")
    print("─────────────────────────────────────────────────────────────────────\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(
    streamflow_config: str = "experiments/routed/random/config.yaml",
    output_path: str = "pretrained_runoff_eval.nc",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    n_catchments: int = 1,  # use the N smallest outlets by upstream area
) -> None:
    if sys.argv[0].endswith(("eval_pretrained_runoff_net.py", "__main__.py")) and len(sys.argv) > 1:
        a = parse_args()
        streamflow_config = a.streamflow_config
        output_path = a.output_path
        device = a.device

    sf_cfg = load_merged_config(streamflow_config)

    # ── 1. DataModule (routed, HRU-level) ────────────────────────────────────
    print(f"Setting up GraphDataModule from: {streamflow_config}")
    dm_kwargs = data_kwargs_from_config(sf_cfg)
    dm_kwargs.update(batch_size=1, num_workers=0, ds_subset={"time": slice("2000-01-01", "2001-12-31")})
    dm = GraphDataModule(**dm_kwargs)

    # ── 2. StreamflowNet: pretrained RunoffNet + random RoutingModel ──────────
    model_init_args = sf_cfg["model"]["init_args"].copy()

    # For the output: use the norm_args_targets stored in the lumped checkpoint so that
    # scale_output=True correctly un-normalizes (LogXp → mm/day).
    # The routed DM has norm_targets=False → NoNormalizer, which would leave the LSTM
    # output in LogXp space and make convert_mmd_to_m3s operate on wrong values.
    dummy = StreamflowNet.__new__(StreamflowNet)
    ckpt_path = dummy._find_best_runoff_net_ckpt(model_init_args["runoff_net_exp_dir"])
    lumped_norm_args_targets = torch.load(ckpt_path, map_location="cpu")["hyper_parameters"]["norm_args_targets"]
    print(f"Using lumped norm_args_targets from checkpoint: {lumped_norm_args_targets}")

    model_init_args.update(
        num_stat_features=dm.num_stat_features,
        num_stat_routing_features=dm.num_stat_routing_features,
        num_dynamic_features=dm.num_dyn_features,
        prediction_length=dm.prediction_length,
        norm_args_stat_features=dm.norm_args_stat_features,   # HRU-scale input stats ✓
        norm_args_dyn_features=dm.norm_args_dyn_features,     # HRU-scale input stats ✓
        norm_args_targets=lumped_norm_args_targets,            # lumped LogXp → correct mm/day output ✓
        norm_args_stat_routing_features=dm.norm_args_stat_routing_features,
        runoff_use_pretrained=True,
        use_instant_routing=True,  # bypass untrained routing weights
    )

    print(f"Loading StreamflowNet (pretrained RunoffNet from: {model_init_args['runoff_net_exp_dir']})")
    model = StreamflowNet(**model_init_args).to(device).eval()

    # ── 3. Restrict to the N smallest outlets that have observations ──────────
    outlets = dm.pred_catchments.values
    if dm.has_obs is None:
        raise RuntimeError("dm.has_obs is None — DataModule was not fully initialised.")
    has_obs = dm.has_obs.sel(hru=outlets).values
    outlets_with_obs = outlets[has_obs]
    upstream_areas = dm.ds.upstream_area.sel(hru=outlets_with_obs).values
    smallest_hrus = outlets_with_obs[np.argsort(upstream_areas)[:n_catchments]]
    dm.pred_catchments = dm.ds.hru.sel(hru=smallest_hrus)
    print(f"Running on {n_catchments} smallest outlet(s) with obs: {dm.pred_catchments.values.tolist()}")

    # ── 4. Predict dataloader ─────────────────────────────────────────────────
    pred_dl = dm.predict_dataloader()

    # ── 5. Run inference ──────────────────────────────────────────────────────
    records = run_inference(model=model, dataloader=pred_dl, device=device)

    # ── 5. Concatenate ────────────────────────────────────────────────────────
    Q_instant_m3s = np.concatenate([r["Q_instant_m3s"] for r in records], axis=0)
    Q_routed_m3s = np.concatenate([r["Q_routed_m3s"] for r in records], axis=0)
    Q_obs_m3s = np.concatenate([r["Q_obs_m3s"] for r in records], axis=0)
    hru_ids = np.concatenate([r["hru_idx"] for r in records], axis=0)
    time_starts = np.concatenate([np.atleast_1d(r["time_start"]) for r in records])

    print_summary(Q_instant_m3s, Q_obs_m3s, "Instantly routed Q vs Q_obs (all outlets x all timesteps)")
    print_summary(Q_routed_m3s, Q_obs_m3s, "Randomly init RoutingModel Q vs Q_obs (all outlets x all timesteps)")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    T = Q_instant_m3s.shape[1]
    ds_out = xr.Dataset(
        {
            "Q_instant_m3s": (
                ["sample", "lead_time"],
                Q_instant_m3s,
                {"units": "m3/s", "long_name": "Instantly routed streamflow (area-weighted, no routing weights)"},
            ),
            "Q_routed_m3s": (
                ["sample", "lead_time"],
                Q_routed_m3s,
                {"units": "m3/s", "long_name": "Streamflow from randomly initialised RoutingModel"},
            ),
            "Q_obs_m3s": (
                ["sample", "lead_time"],
                Q_obs_m3s,
                {"units": "m3/s", "long_name": "Observed streamflow"},
            ),
            "hru": (["sample"], hru_ids, {"long_name": "HRU index"}),
            "time_start": (["sample"], time_starts, {"long_name": "First timestep index"}),
        },
        coords={"lead_time": np.arange(T)},
        attrs={
            "description": (
                "StreamflowNet forward pass with pretrained RunoffNet. "
                "Q_instant_m3s uses area-weighted aggregation (no routing weights); "
                "Q_routed_m3s uses the randomly initialised RoutingModel."
            ),
            "streamflow_config": str(streamflow_config),
            "runoff_net_exp_dir": str(model_init_args["runoff_net_exp_dir"]),
        },
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
