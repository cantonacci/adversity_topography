#!/bin/bash
# =============================================================================
# submit_fc_followup.sh
#
# Submit FC compute array for a single follow-up session.
# Run once per timepoint; all three can be submitted in parallel.
#
# Usage:
#   sbatch submit_fc_followup.sh ses-02A
#   sbatch submit_fc_followup.sh ses-04A
#   sbatch submit_fc_followup.sh ses-06A
#
# Outputs land in:
#   outputs/data_processed/fc_chunks_{session}/fc_chunk_*.csv
# =============================================================================

#SBATCH --job-name=fc_followup
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-19
#SBATCH --output=outputs/logs/fc_%A_%a.out
#SBATCH --error=outputs/logs/fc_%A_%a.err

set -euo pipefail

SESSION=${1}

if [ -z "$SESSION" ]; then
    echo "ERROR: must pass session as first argument (e.g. ses-02A)"
    exit 1
fi


[ -f config.local.sh ] && source config.local.sh
PYTHON=${PYTHON:-python3}
CODE=code

echo "======================================================"
echo "Job ID   : $SLURM_JOB_ID  array task: $SLURM_ARRAY_TASK_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Session  : $SESSION"
echo "Start    : $(date)"
echo "======================================================"

$PYTHON "$CODE/03_connectivity/compute_fc.py" \
    --session "$SESSION" \
    --chunk-idx "$SLURM_ARRAY_TASK_ID" \
    --n-chunks 20

echo "Done: $(date)"
