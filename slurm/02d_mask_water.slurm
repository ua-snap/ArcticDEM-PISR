#!/bin/bash
#SBATCH --job-name=adem-water
#SBATCH --partition=analysis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --output=/beegfs/CMIP6/jdpaul3/ArcticDEM-PISR_data/logs/%x_%j.out
# Set PISR to NoData over water (ArcticDEM flattened lakes/sea). Run after 02c. Submit from repo root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
RUN="${MAMBA_EXE:-$HOME/.local/bin/micromamba} run -n arcticdem-pisr python -I"
/usr/bin/time -v $RUN scripts/02_pisr/mask_water.py
