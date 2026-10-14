#!/bin/bash
#SBATCH --job-name=adem-warp3338
#SBATCH --partition=analysis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/beegfs/CMIP6/jdpaul3/ArcticDEM-PISR_data/logs/%x_%j.out
# Warp the 3413 DEM VRT to the 32 m EPSG:3338 target grid. Submit from repo root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
RUN="${MAMBA_EXE:-$HOME/.local/bin/micromamba} run -n arcticdem-pisr python -I"
/usr/bin/time -v $RUN scripts/02_pisr/warp_dem.py
