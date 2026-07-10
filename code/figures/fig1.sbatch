#!/bin/bash
#SBATCH -J fig1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o outputs/logs/fig1_%j.out
#SBATCH -e outputs/logs/fig1_%j.err

set -euo pipefail
cd code/figures
[ -f config.local.sh ] && source config.local.sh
PYTHON=${PYTHON:-python3}
mkdir -p outputs/logs
echo "Started $(date) on $(hostname)"
${PYTHON} fig1_panels.py
echo "Done $(date)"
