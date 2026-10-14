#!/bin/bash
#SBATCH --job-name=adem-download
#SBATCH --partition=transfer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/beegfs/CMIP6/jdpaul3/ArcticDEM-PISR_data/logs/%x_%j.out
# Downloads all ArcticDEM tiles in the tile list (resumable), then validates and builds the VRTs.
# Submit from the repo root:  sbatch slurm/01_download.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
RUN="${MAMBA_EXE:-$HOME/.local/bin/micromamba} run -n arcticdem-pisr python -I"
$RUN scripts/01_download/download_tiles.py
$RUN scripts/01_download/validate_tiles.py --workers 4
