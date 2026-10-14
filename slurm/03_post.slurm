#!/bin/bash
#SBATCH --job-name=adem-post
#SBATCH --partition=analysis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/beegfs/CMIP6/jdpaul3/ArcticDEM-PISR_data/logs/%x_%j.out
# Phase 3: QC (+ overviews) -> figures -> CF netCDF. Submit from repo root, e.g. after the PISR run:
#   sbatch --dependency=afterok:<pisr_jobid> slurm/03_post.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
RUN="${MAMBA_EXE:-$HOME/.local/bin/micromamba} run -n arcticdem-pisr python -I"
$RUN scripts/03_post/qc.py
$RUN scripts/03_post/figures.py
/usr/bin/time -v $RUN scripts/03_post/to_netcdf.py
