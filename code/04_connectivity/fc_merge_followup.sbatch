#!/bin/bash
# =============================================================================
# merge_fc_followup.sh
#
# Merge per-chunk CSVs from compute_fc.py into a single fc_{session}.csv.
# Run after all array tasks complete (or submit with --dependency).
#
# Usage:
#   sbatch --dependency=afterok:<array_job_id> merge_fc_followup.sh ses-02A
#   # or manually after confirming all chunks exist:
#   bash merge_fc_followup.sh ses-02A
# =============================================================================

#SBATCH --job-name=fc_merge
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/fcmerge_%j.out
#SBATCH --error=outputs/logs/fcmerge_%j.err

set -euo pipefail

SESSION=${1}

if [ -z "$SESSION" ]; then
    echo "ERROR: must pass session as first argument (e.g. ses-02A)"
    exit 1
fi


[ -f config.local.sh ] && source config.local.sh
PYTHON=${PYTHON:-python3}
DAT=outputs/data_processed
CHUNK_DIR="$DAT/fc_chunks_${SESSION}"
OUT_FILE="$DAT/fc_${SESSION}.csv"

echo "======================================================"
echo "Merging FC chunks for session: $SESSION"
echo "Chunk dir : $CHUNK_DIR"
echo "Output    : $OUT_FILE"
echo "Start     : $(date)"
echo "======================================================"

N_CHUNKS=$(ls "$CHUNK_DIR"/fc_chunk_*.csv 2>/dev/null | wc -l)
echo "Found $N_CHUNKS chunk files"

if [ "$N_CHUNKS" -eq 0 ]; then
    echo "ERROR: no chunk files found in $CHUNK_DIR"
    exit 1
fi

export SESSION
$PYTHON - <<'PYEOF'
import sys, os, glob
import pandas as pd

session   = os.environ['SESSION']
dat       = 'outputs/data_processed'
chunk_dir = f'{dat}/fc_chunks_{session}'
out_file  = f'{dat}/fc_{session}.csv'

chunks = sorted(glob.glob(f'{chunk_dir}/fc_chunk_*.csv'))
print(f'Reading {len(chunks)} chunks...')
dfs = [pd.read_csv(c) for c in chunks]
df  = pd.concat(dfs, ignore_index=True)

print(f'Total rows : {len(df)}')
print(f'Subjects   : {df["sub_ID"].nunique()}')

n_fd_note = (df['n_usable_frames'] < 375).sum()
if n_fd_note > 0:
    print(f'WARNING: {n_fd_note} rows have n_usable_frames < 375 (should not happen)')

df.to_csv(out_file, index=False)
print(f'Saved → {out_file}')
PYEOF

echo "Done: $(date)"
