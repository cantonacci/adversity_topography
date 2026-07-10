#!/bin/bash
# Template for machine-specific configuration.
#
#   cp config.local.example.sh config.local.sh   # then edit for your machine
#
# config.local.sh is git-ignored and is read by both the shell launchers
# (run_all.sh, code/**/*.sbatch) and by code/config.py. Any variable may
# alternatively be set as an environment variable.

# Python interpreter, plus any environment/module setup your system needs.
# On an HPC you might add, e.g.:  module load python/3.12
export PYTHON=python3

# External derivative locations (large inputs that are NOT stored in this repo).
export REPRO_DIR=/path/to/network_parcellations     # per-vertex boldmap .dlabel.nii files
export XCP_DIR=/path/to/anatomical_surfaces         # midthickness .surf.gii files
export FC_DTSERIES_DIR=/path/to/rest_timeseries      # preprocessed dense timeseries
export AB_G_DYN_FILE=/path/to/ab_g_dyn.tsv           # dynamic covariates (year-2 visit age)
export AB_COVARIATES_DIR=/path/to/covariate_extracts # raw covariate tables
