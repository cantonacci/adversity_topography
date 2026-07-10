#!/bin/bash
#SBATCH --job-name=med_rerun
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o outputs/logs/med_rerun_%j.out
#SBATCH -e outputs/logs/med_rerun_%j.err
# Submit from repo root:  sbatch -p <partition> code/05_behavior/mediation.sbatch
set -uo pipefail
# Under sbatch, BASH_SOURCE points at the Slurm spool copy, not the repo — use
# the submission directory (submit this script from the repository root).
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
[ -f config.local.sh ] && source config.local.sh
PYTHON=${PYTHON:-python3}
mkdir -p outputs/logs
echo "== mediation re-run $(date) on $(hostname) =="
$PYTHON code/05_behavior/mediation.py
echo "== done rc=$? $(date) =="
