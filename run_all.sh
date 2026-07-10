#!/bin/bash
# =============================================================================
# run_all.sh — master pipeline driver for the ELA × SCAN-topography analyses.
#
# The repo is organized to mirror the manuscript:
#   code/00_surface_area network parcellations -> per-network cortical-area CSVs
#   code/01_data_prep    build the analysis dataframes (df_base/y2/y4/y6, composites)
#   code/02_expansion    selective SCAN expansion (bivariate, multivariate, split-half, sibling)
#   code/03_encroachment SCAN territory displacement + gradient positioning
#   code/04_connectivity SCAN functional-connectivity shift + structure-function correspondence
#   code/05_behavior     cognitive cost (mediation, by-wave, prediction, within-person)
#   code/tables          Table 1 / participant characteristics
#   code/figures         Figures 1-4 panels + surface/CIFTI generators
#   code/supplement      convergence diagnostic, etc.
#
# Two layers:
#   HEAVY (external neuroimaging inputs; run once via array jobs):
#     - Network-area CSVs from parcellations   (00_surface_area/compute_network_areas.py)
#     - FC matrices from preprocessed dtseries (04_connectivity/compute_fc*.py)
#     - Encroachment fractions from parcellations (03_encroachment/compute_encroachment.py)
#     These read data NOT in this repo (see README "External data"). They are the
#     slow, one-time steps and are SKIPPED by default (SKIP_HEAVY=1).
#   ANALYSIS + FIGURES (reproducible from the processed dataframes / intermediate
#     FC & encroachment CSVs already in outputs/). This is what run_all.sh runs.
#
# Usage:
#   sbatch run_all.sh                 # analysis+figure layer (SKIP_HEAVY=1 default)
#   SKIP_HEAVY=0 sbatch run_all.sh    # also recompute FC matrices + encroachment (needs external data)
#   bash run_all.sh --stages A,B,E    # run only selected stages (interactive/sh_dev)
#
# =============================================================================
#SBATCH --job-name=adv_topo_all
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o outputs/logs/run_all_%j.out
#SBATCH -e outputs/logs/run_all_%j.err
# Submit from the repository root, choosing a partition for your scheduler:
#   sbatch -p <partition> run_all.sh

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$ROOT/code"
SKIP_HEAVY=${SKIP_HEAVY:-1}

# Environment setup (Python interpreter, any module loads) is machine-specific:
# put it in config.local.sh (copy from config.local.example.sh). Falls back to
# `python3` on PATH.
[ -f "$ROOT/config.local.sh" ] && source "$ROOT/config.local.sh"
PYTHON=${PYTHON:-python3}
mkdir -p "$ROOT/outputs/logs"
cd "$ROOT"

# Optional stage selection: --stages A,B,C ...  (default: all)
STAGES="A B C D E F G"
if [ "${1:-}" == "--stages" ] && [ -n "${2:-}" ]; then
    STAGES=$(echo "$2" | tr ',' ' ')
fi
want() { echo " $STAGES " | grep -q " $1 "; }

run() {  # run <relative-script> [args...]
    echo ""; echo "── $(date '+%H:%M:%S')  $1 ${*:2}"
    $PYTHON "$CODE/$1" "${@:2}"
    local rc=$?
    if [ $rc -ne 0 ]; then echo "ERROR: $1 exited $rc — stopping."; exit $rc; fi
}

echo "==== run_all.sh  SKIP_HEAVY=$SKIP_HEAVY  stages=[$STAGES]  $(date) ===="

# ── Stage 0: surface-area extraction (HEAVY: needs external parcellations) ─────
# Produces data/network_areas/{00A,02A,04A,06A}_network_areas_cortical.csv, which
# Stage A reads. Skipped by default; the CSVs ship as the pipeline's entry point.
if [ "$SKIP_HEAVY" -eq 0 ]; then
  for SES in ses-00A ses-02A ses-04A ses-06A; do
    run 00_surface_area/compute_network_areas.py --session "$SES"
  done
else
  echo "  (skip) 00_surface_area/compute_network_areas.py  [SKIP_HEAVY=1; uses existing data/network_areas/*.csv]"
fi

# ── Stage A: data preparation ────────────────────────────────────────────────
if want A; then
  run 01_data_prep/build_ela_composites.py
  run 01_data_prep/build_analysis_dataframes.py
  run 01_data_prep/fix_y2_age.py            # MUST follow build_analysis_dataframes (year-2 age fix)
  run 01_data_prep/add_behavioral_outcomes.py
fi

# ── Stage B: expansion ───────────────────────────────────────────────────────
if want B; then
  run 02_expansion/bivariate_associations.py
  run 02_expansion/multivariate_models.py
  run 02_expansion/splithalf_replication.py
  run 02_expansion/sibling_discordance.py
fi

# ── Stage C: encroachment ────────────────────────────────────────────────────
if want C; then
  if [ "$SKIP_HEAVY" -eq 0 ]; then
    run 03_encroachment/compute_encroachment.py     # HEAVY: needs ReproTM boldmaps
  else
    echo "  (skip) 03_encroachment/compute_encroachment.py  [SKIP_HEAVY=1; uses existing encroachment_*.csv]"
  fi
  run 03_encroachment/encroachment_analysis.py
  run 03_encroachment/gradient_positioning.py
  run 03_encroachment/encroachment_cifti.py
fi

# ── Stage D: functional connectivity ─────────────────────────────────────────
if want D; then
  if [ "$SKIP_HEAVY" -eq 0 ]; then
    echo "  NOTE: FC matrices are computed via array jobs, not inline. Submit separately:"
    echo "    ARR=\$(sbatch --parsable code/03_connectivity/fc_compute_baseline.sbatch)"
    echo "    sbatch --dependency=afterok:\$ARR ... then fc_analysis below. Skipping inline recompute."
  else
    echo "  (skip) 03_connectivity/compute_fc*.py  [SKIP_HEAVY=1; uses existing fc_ses-*.csv]"
  fi
  run 04_connectivity/fc_analysis.py
  run 04_connectivity/encroach_fc_correspondence.py
  run 04_connectivity/fc_figures.py
fi

# ── Stage E: behavior (cognitive cost) ───────────────────────────────────────
if want E; then
  run 05_behavior/mediation.py
  run 05_behavior/cognition_by_wave.py
  run 05_behavior/cv_prediction.py
  run 05_behavior/within_person/build_within_person.py
  run 05_behavior/within_person/within_person_cognition.py
  run 05_behavior/within_person/within_person_cbcl.py     # supplement (null)
fi

# ── Stage F: tables ──────────────────────────────────────────────────────────
if want F; then
  run tables/table1_sample_characteristics.py
  run tables/participant_characteristics.py
fi

# ── Stage G: figure-data + panels ────────────────────────────────────────────
if want G; then
  run figures/compute_network_gradient.py
  run figures/compute_gradient_hotspot.py
  run figures/prep_fig4_withinperson.py
  run figures/fig1_panels.py
  run figures/fig2_panels.py
  run figures/fig3_panels.py
  run figures/fig4_panels.py
fi

echo ""; echo "==== run_all.sh complete  $(date) ===="
