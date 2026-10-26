# Cluster-specific SLURM settings: template.
#
# Copy this file to slurm/cluster.sh (not tracked by git) and adjust it to your cluster. It is loaded automatically
# by all scripts in slurm/, run_all.sh and experiments/*/run.sh, both when submitting and inside the jobs.

# Account to charge jobs to, read by sbatch. Remove the line if your cluster does not use accounts.
export SBATCH_ACCOUNT=your_account

# GPU request for training jobs, passed as `--gpus`, e.g. "1" or "nvidia_geforce_rtx_4090:1".
GPU_SPEC=1

# Partition for CPU-only jobs (combining predictions, source contributions). Leave empty for the cluster default.
CPU_PARTITION=

# Runs at the start of every job, e.g. to load modules.
cluster_setup() {
    # module load cuda/12.8.0
    :
}
