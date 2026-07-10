#!/bin/bash
#SBATCH -J fc_compute
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --array=0-19
#SBATCH -o outputs/logs/fc_compute_%A_%a.out
#SBATCH -e outputs/logs/fc_compute_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"

[ -f config.local.sh ] && source config.local.sh
PYTHON=${PYTHON:-python3}

echo "===================================="
echo "FC compute  array=${SLURM_ARRAY_TASK_ID}  started: $(date)"
echo "Node: $(hostname)"
echo "===================================="

mkdir -p outputs/logs
mkdir -p outputs/data_processed/fc_chunks

${PYTHON} code/03_connectivity/compute_fc_baseline.py \
    --chunk-idx ${SLURM_ARRAY_TASK_ID} \
    --n-chunks 20

echo "Done: $(date)"
