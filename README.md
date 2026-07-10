# Early Adversity Selectively Reshapes the Somato-Cognitive Action Network

Analysis code for "Early Adversity Selectively Reshapes the Somato-Cognitive Action Network in the Developing Brain" (Antonacci et al.; manuscript under review).

The code is organized to mirror the manuscript: surface-area extraction and data
preparation build the analysis dataframes, which feed four analysis sections —
**expansion → encroachment → connectivity → behavior** — plus figures, tables,
and a supplement. Only code and configuration are distributed here; the
ABCD-restricted data, generated outputs, and the manuscript itself are
git-ignored and are not part of this repository.

---

## Background

Network topography — the cortical surface area occupied by each functional
network — is a stable, heritable individual-differences variable. Because total
cortical area is roughly conserved, network areas are zero-sum: greater
allocation to one network implies less to others.

Early-life adversity (ELA) is operationalized using 10 data-driven dimensions of
adversity co-occurrence identified in the ABCD baseline sample by Brieant et al.
(2023), aggregated into three a priori composites (threat, deprivation,
unpredictability). Individualized network assignments come from the ReproTM
Template Matching pipeline (Hermosillo et al., 2024; template ABCC-a3-9to16).

**Headline result.** The SCAN (Somato-Cognitive Action Network) shows the
strongest, most selective association with adversity: under higher threat, SCAN
occupies a larger share of cortex, encroaching primarily on somatomotor and
cingulo-opercular territory; its functional connectivity shifts toward
sensorimotor/auditory networks and away from salience/control networks; and a
larger SCAN carries a measurable cost to crystallized cognition.

---

## Repository structure

```
adversity_topography/
├── run_all.sh                # master pipeline driver
├── requirements.txt          # pinned Python environment (pip freeze)
├── config.local.example.sh   # template for machine-specific paths (copy to config.local.sh)
├── code/
│   ├── config.py                     # paths, network list, ELA composites, FDR counts
│   ├── 00_surface_area/              # topography → per-network cortical-area CSVs
│   │   └── compute_network_areas.py      # network areas from parcellations + surfaces (needs derivatives)
│   ├── 01_data_prep/                 # build the analysis dataframes
│   │   ├── build_ela_composites.py       # threat/deprivation/unpredictability composites
│   │   ├── build_analysis_dataframes.py  # df_base / df_y2 / df_y4 / df_y6 (topography + covariates)
│   │   ├── fix_y2_age.py                  # year-2 age correction (run AFTER build_analysis_dataframes)
│   │   ├── add_behavioral_outcomes.py     # merge NIH Toolbox + CBCL
│   │   └── build_covariates.py            # extended covariate assembly
│   ├── 02_expansion/                 # selective SCAN expansion
│   │   ├── bivariate_associations.py      # 10 ELA × 15 network partial correlations
│   │   ├── multivariate_models.py         # multivariate mixed models + ΔR² selectivity
│   │   ├── splithalf_replication.py       # discovery/replication split-half
│   │   └── sibling_discordance.py         # within-family (sibling) estimates
│   ├── 03_encroachment/              # SCAN territory displacement
│   │   ├── compute_encroachment.py        # per-subject encroachment fractions (needs derivatives)
│   │   ├── encroachment_analysis.py       # zone dissociation statistics + figures
│   │   ├── gradient_positioning.py        # expansion vs principal cortical gradient (spin test)
│   │   ├── encroachment_cifti.py          # CIFTI maps for Workbench
│   │   └── generate_scan_borders.py       # SCAN dlabel + border files
│   ├── 04_connectivity/              # SCAN functional-connectivity shift
│   │   ├── compute_fc_baseline.py         # baseline 15-network FC matrices (array; needs dtseries)
│   │   ├── compute_fc.py                  # follow-up sessions FC matrices (array)
│   │   ├── fc_analysis.py                 # threat → SCAN-FC LME (dual FDR)
│   │   ├── encroach_fc_correspondence.py  # structure–function correspondence
│   │   ├── seed_fc.py                     # SCAN seed-FC group-mean surfaces
│   │   └── fc_figures.py                  # FC figures
│   ├── 05_behavior/                  # cognitive cost
│   │   ├── mediation.py                   # threat → SCAN → cognition (+ CBCL supplement)
│   │   ├── cognition_by_wave.py           # cross-sectional cognition mediation per wave
│   │   ├── cv_prediction.py               # cross-validated prediction of y6 crystallized
│   │   └── within_person/                 # within-person change-score models (+ derived/ data)
│   │       ├── build_within_person.py
│   │       ├── within_person_cognition.py
│   │       ├── within_person_cbcl.py      # supplement (null)
│   │       └── figure_within_person.py
│   ├── tables/                       # Table 1 / participant characteristics
│   ├── figures/                      # Figure 1–4 panels + surface/CIFTI generators
│   │                                 #   figstyle.py + nature.mplstyle = house style; figsrc.py = project style
│   └── supplement/                   # Bayes factors, RE-specification robustness, etc.
│
│   # ── git-ignored (local only; NOT in the public repository) ────────────────
├── data/                             # ABCD-restricted inputs (see "External data")
├── outputs/                          # generated: data_processed/, tables/, figures/, cifti_for_workbench/
├── manuscript/                       # the paper and its assets
└── config.local.sh                   # your machine's real paths (copied from config.local.example.sh)
```

Everything below the marker above is git-ignored: cloning the repository yields
only `code/`, `run_all.sh`, `config.local.example.sh`, `requirements.txt`,
`LICENSE`, and this README. Restricted ABCD data and the manuscript are not distributed.

Output CSV filenames retain their original `phase*`/analysis-letter stems
(e.g. `phase2_composites_r_matrix_baseline.csv`, `fc_lme_threat_baseline.csv`) —
these are the data contract read by the figure scripts and are intentionally
unchanged from prior runs.

---

## Environment

Python 3.12; exact package versions are pinned in `requirements.txt`
(numpy 2.2.6, pandas 2.3.2, statsmodels 0.14.6, scikit-learn 1.7.2,
nibabel 5.4.0, neuromaps 0.0.7, …):

```bash
python -m venv env && source env/bin/activate && pip install -r requirements.txt
```

Machine-specific settings — the Python interpreter, any module loads, and the
locations of the external derivatives below — go in `config.local.sh`:

```bash
cp config.local.example.sh config.local.sh   # then edit for your machine
```

`config.local.sh` is git-ignored and is read by both the launchers (`run_all.sh`,
`code/**/*.sbatch`) and `code/config.py`. All scripts locate `config.py` by
walking up to the `code/` root, so they run correctly from any working directory.

---

## External data (not contained in this repository)

ABCD data — including the ABCC (ABCD-BIDS Community Collection) derivatives used
here — are distributed through the NIH Brain Development Cohorts (NBDC) Data Hub
(https://www.nbdc-datahub.org/) and require an approved Data Use Agreement to
download. Analyses used ABCD Release 7.0.

The processed dataframes under `outputs/data_processed/` are sufficient to
reproduce every reported statistic and figure. Regenerating them from raw inputs
additionally requires the following ABCC/ABCD derivatives, whose locations are set
in `config.local.sh`:

| `config.local.sh` variable | Download / contents | Used by |
|---|---|---|
| `REPRO_DIR` | ReproTM template-matching topography (per-vertex `boldmap` `.dlabel.nii`) | surface area, encroachment |
| `XCP_DIR` | anatomical midthickness surfaces (`fsLR_den-32k_desc-hcp_midthickness.surf.gii`) | surface area, encroachment/CIFTI |
| `FC_DTSERIES_DIR` | XCP-D–postprocessed resting-state dense timeseries (`.dtseries.nii`) | FC matrices |
| `AB_COVARIATES_DIR` | motion/QC (framewise displacement) and demographic covariates (age, sex, study site, family, race/income/education) | covariate assembly |
| `AB_G_DYN_FILE` | year-2 visit-age table | `fix_y2_age.py` |

The ReproTM atlas dlabel ships under `data/atlas_files/`.

---

## Running the analyses

Run everything from the repository root (submit to your scheduler with the
appropriate partition, e.g. `sbatch -p <partition> run_all.sh`):

```bash
# Full analysis + figure layer from the processed dataframes (default):
sbatch run_all.sh

# Also regenerate the heavy inputs (network-area CSVs, FC matrices, encroachment)
# from the external derivatives set in config.local.sh:
SKIP_HEAVY=0 sbatch run_all.sh

# Selected stages only (A=data_prep B=expansion C=encroachment
#                       D=connectivity E=behavior F=tables G=figures):
bash run_all.sh --stages B,E
```

The one-time heavy steps — network areas from parcellations, FC matrices from
dtseries, encroachment from parcellations — read the external derivatives and are
skipped by default; `run_all.sh` then reads the shipped `data/network_areas/*.csv`
and the existing `fc_ses-*.csv` / `encroachment_*.csv` intermediates. Run order
and dependencies are documented inline in `run_all.sh`.

---

## Key covariate / modeling notes

| Variable | Role |
|---|---|
| `interview_age`, `sex_num` | fixed covariates |
| `rest_mean_FD_[wave]` | fixed covariate (topography); FC analyses add `mean_FD`, `n_usable_frames` |
| `study_site` | 21 ABCD research sites (22 indicator levels incl. one administrative level; scanner/manufacturer nested within site) |
| `family_id` | sibling grouping (families do not span sites) |
| `Race`, `Income`, `Parent_edu` | sensitivity only (overlap with ELA) |

Precise model specifications are given in the manuscript Methods.

---

## References

- Brieant et al. (2023). *Dev Cogn Neurosci*, 61, 101256.
- Hermosillo et al. (2024). *Nature Neuroscience*, 27, 1000–1013.
- Lynch et al. (2024). *Nature*, 633, 624–633.
- Volkow et al. (2018). *Dev Cogn Neurosci*, 32, 4–7.

---

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.
