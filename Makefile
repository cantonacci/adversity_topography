# =============================================================================
# Makefile — idempotent workflow manager for the ELA × SCAN-topography analyses.
#
# Mirrors the stages in run_all.sh but with real file targets, so re-running
# `make` only re-executes steps whose inputs changed. Stage aggregates are
# phony; the per-step targets are the key output files each script writes.
#
# Usage:
#   make                 # build everything reproducible from processed inputs
#   make dataprep        # just the analysis dataframes
#   make expansion       # SCAN-expansion analyses (needs dataprep)
#   make figures         # figure panels (needs upstream stages)
#   make test            # run the pytest suite
#   make clean           # remove derived tables/figures (NOT processed data)
#   make -n <target>     # dry-run: show what would rebuild
#
# The Python interpreter and external paths come from config.local.sh (copy it
# from config.local.example.sh). Override with:  make PYTHON=/path/to/python
# =============================================================================

# Resolve PYTHON from config.local.sh, then the environment, then python3.
PYTHON ?= $(shell sh -c '. ./config.local.sh >/dev/null 2>&1; echo $${PYTHON:-python3}')

CODE := code
OUT  := outputs
TAB  := $(OUT)/tables
FIG  := $(OUT)/figures
DAT  := $(OUT)/data_processed

# Run a script, echoing a timestamped banner (mirrors run_all.sh's `run`).
define run
	@echo ""; echo "── $$(date '+%H:%M:%S')  $(1)"
	$(PYTHON) $(CODE)/$(1)
endef

.PHONY: all dataprep expansion encroachment connectivity behavior tables figures test clean help

all: figures tables ## Build the full reproducible analysis + figure layer

# ── Stage A: data preparation ────────────────────────────────────────────────
# fix_y2_age.py MUST follow build_analysis_dataframes.py (year-2 age correction),
# and add_behavioral_outcomes.py appends NIH/CBCL outcomes. df_base.csv is the
# sentinel for the whole stage.
DFRAMES := $(DAT)/df_base.csv $(DAT)/df_y2.csv $(DAT)/df_y4.csv $(DAT)/df_y6.csv

$(DAT)/df_base.csv: \
		$(CODE)/01_data_prep/build_ela_composites.py \
		$(CODE)/01_data_prep/build_analysis_dataframes.py \
		$(CODE)/01_data_prep/fix_y2_age.py \
		$(CODE)/01_data_prep/add_behavioral_outcomes.py \
		$(CODE)/config.py
	$(call run,01_data_prep/build_ela_composites.py)
	$(call run,01_data_prep/build_analysis_dataframes.py)
	$(call run,01_data_prep/fix_y2_age.py)
	$(call run,01_data_prep/add_behavioral_outcomes.py)

dataprep: $(DAT)/df_base.csv ## Build analysis dataframes (df_base/y2/y4/y6)

# ── Stage B: expansion ───────────────────────────────────────────────────────
$(TAB)/phase3_composites_delta_r2_baseline.csv: \
		$(CODE)/02_expansion/multivariate_models.py $(DFRAMES)
	$(call run,02_expansion/bivariate_associations.py)
	$(call run,02_expansion/multivariate_models.py)
	$(call run,02_expansion/splithalf_replication.py)
	$(call run,02_expansion/sibling_discordance.py)

expansion: $(TAB)/phase3_composites_delta_r2_baseline.csv ## SCAN-expansion analyses

# ── Stage C: encroachment ────────────────────────────────────────────────────
$(TAB)/encroachment_regression_baseline.csv: \
		$(CODE)/03_encroachment/encroachment_analysis.py $(DFRAMES)
	$(call run,03_encroachment/encroachment_analysis.py)
	$(call run,03_encroachment/gradient_positioning.py)
	$(call run,03_encroachment/encroachment_cifti.py)

encroachment: $(TAB)/encroachment_regression_baseline.csv ## SCAN territory displacement

# ── Stage D: functional connectivity ─────────────────────────────────────────
connectivity: dataprep ## FC shift + structure-function (needs precomputed FC CSVs)
	$(call run,04_connectivity/fc_analysis.py)
	$(call run,04_connectivity/encroach_fc_correspondence.py)
	$(call run,04_connectivity/fc_figures.py)

# ── Stage E: behavior (cognitive cost) ───────────────────────────────────────
behavior: dataprep ## Mediation, by-wave, CV prediction, within-person
	$(call run,05_behavior/mediation.py)
	$(call run,05_behavior/cognition_by_wave.py)
	$(call run,05_behavior/cv_prediction.py)
	$(call run,05_behavior/within_person/build_within_person.py)
	$(call run,05_behavior/within_person/within_person_cognition.py)
	$(call run,05_behavior/within_person/within_person_cbcl.py)

# ── Stage F: tables ──────────────────────────────────────────────────────────
tables: dataprep ## Table 1 / participant characteristics
	$(call run,tables/table1_sample_characteristics.py)
	$(call run,tables/participant_characteristics.py)

# ── Stage G: figure-data + panels ────────────────────────────────────────────
figures: expansion encroachment behavior ## Figure 1–4 panels (needs upstream stages)
	$(call run,figures/compute_network_gradient.py)
	$(call run,figures/compute_gradient_hotspot.py)
	$(call run,figures/prep_fig4_withinperson.py)
	$(call run,figures/fig1_panels.py)
	$(call run,figures/fig2_panels.py)
	$(call run,figures/fig3_panels.py)
	$(call run,figures/fig4_panels.py)

# ── Tests ────────────────────────────────────────────────────────────────────
test: ## Run the pytest suite (data-mutation accuracy + idempotency, etc.)
	$(PYTHON) -m pytest -q $(CODE)/tests

# ── Housekeeping ─────────────────────────────────────────────────────────────
clean: ## Remove derived tables and figures (leaves processed dataframes intact)
	rm -f $(TAB)/*.csv $(TAB)/*.txt $(FIG)/*.png
	@echo "Cleaned derived tables + figures. Processed dataframes in $(DAT) kept."

help: ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
