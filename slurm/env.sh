# Loads the cluster-specific SLURM settings from slurm/cluster.sh (not tracked by git).
#
# Sourced by every submit and job script; all of them are run from the repository root, which SLURM also uses as the
# working directory of the jobs.

if [ ! -f slurm/cluster.sh ]; then
    echo "slurm/cluster.sh not found. Copy slurm/cluster.example.sh to slurm/cluster.sh and fill in your cluster" \
        "settings, and run the scripts from the repository root." >&2
    exit 1
fi

source slurm/cluster.sh
