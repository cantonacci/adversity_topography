#!/bin/bash
# =============================================================================
# submit_network_areas.sh
#
# SLURM array job: compute network surface areas for all 4 ABCD timepoints.
# Submits 4 tasks in parallel (one per session), each using 16 CPUs.
#
# Usage:
#   sbatch submit_network_areas.sh
#
# Outputs go to data/network_areas/ (per-session cortical-area CSVs).
#   00A_network_areas_cortical.csv
#   00A_network_areas_with_subcortical.csv
#   02A_network_areas_cortical.csv
#   02A_network_areas_with_subcortical.csv
#   04A_network_areas_cortical.csv
#   04A_network_areas_with_subcortical.csv
#   06A_network_areas_cortical.csv
#   06A_network_areas_with_subcortical.csv
#   (plus one *_failures.txt per session if any subjects could not be processed)
# =============================================================================

#SBATCH --job-name=network_areas
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --output=outputs/logs/network_areas_%a.out
#SBATCH --error=outputs/logs/network_areas_%a.err

# ---- environment -------------------------------------------------------------

[ -f config.local.sh ] && source config.local.sh
PYTHON=${PYTHON:-python3}
SCRIPT=code/00_surface_area/compute_network_areas.py

# ---- session lookup ----------------------------------------------------------
SESSIONS=(ses-00A ses-02A ses-04A ses-06A)
SESSION=${SESSIONS[$SLURM_ARRAY_TASK_ID]}

echo "======================================================"
echo "SLURM array task : $SLURM_ARRAY_TASK_ID"
echo "Session           : $SESSION"
echo "CPUs allocated    : $SLURM_CPUS_PER_TASK"
echo "Node              : $SLURMD_NODENAME"
echo "Start time        : $(date)"
echo "======================================================"

# ---- run ---------------------------------------------------------------------
$PYTHON $SCRIPT \
    --session "$SESSION" \
    --n_jobs "$SLURM_CPUS_PER_TASK"

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"
