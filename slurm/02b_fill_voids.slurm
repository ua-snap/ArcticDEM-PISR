#!/bin/bash
#SBATCH --job-name=adem-fill
#SBATCH --partition=analysis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/beegfs/CMIP6/jdpaul3/ArcticDEM-PISR_data/logs/%x_%j.out
# Fill small enclosed voids in the 32 m EPSG:3338 DEM (whole raster in memory, ~80 GB). Submit from repo root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
RUN="${MAMBA_EXE:-$HOME/.local/bin/micromamba} run -n arcticdem-pisr python -I"
/usr/bin/time -v $RUN scripts/02_pisr/fill_voids.py
