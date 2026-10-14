#!/bin/bash
#SBATCH --job-name=adem-pisr
#SBATCH --partition=analysis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/beegfs/CMIP6/jdpaul3/ArcticDEM-PISR_data/logs/%x_%j.out
# Full PISR run on the 32 m EPSG:3338 DEM. Restartable (finished bands are skipped). Submit from repo root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
RUN="${MAMBA_EXE:-$HOME/.local/bin/micromamba} run -n arcticdem-pisr python -I"
/usr/bin/time -v $RUN scripts/02_pisr/run_pisr.py "$@"
