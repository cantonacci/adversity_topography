"""
Shared paths and constants for all analysis phases.
adversity_topography/code/config.py
"""
from pathlib import Path
import os
import re
import numpy as np

# ── Root directories (relative to the repository; no absolute paths) ──────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / 'data'
CODE_DIR  = BASE_DIR / 'code'
OUT_DIR   = BASE_DIR / 'outputs'
FIG_DIR   = OUT_DIR  / 'figures'
TAB_DIR   = OUT_DIR  / 'tables'
DAT_DIR   = OUT_DIR  / 'data_processed'

# ── Environment-specific derivative locations ─────────────────────────────────
# Large inputs (network parcellations, anatomical surfaces, preprocessed
# timeseries) live outside the repository and differ by machine. Set their
# locations in `config.local.sh` (copy it from `config.local.example.sh`; this
# file is git-ignored). Values may also be supplied as environment variables.
# Generic in-repo defaults are used if neither is provided.
_LOCAL = BASE_DIR / 'config.local.sh'
_local = {}
if _LOCAL.exists():
    for _ln in _LOCAL.read_text().splitlines():
        _m = re.match(r'\s*export\s+(\w+)\s*=\s*(.*)', _ln)
        if _m:
            _local[_m.group(1)] = _m.group(2).strip().strip('"').strip("'")


def _ext(key, default):
    """Resolve an external path from the environment, then config.local.sh, else a default."""
    return Path(os.environ.get(key) or _local.get(key) or default)


# Per-vertex network parcellations (template-matching boldmaps).
REPRO_DIR       = _ext('REPRO_DIR',       DATA_DIR / 'derivatives' / 'network_parcellations')
# Anatomical midthickness surfaces (.surf.gii) from minimal preprocessing.
XCP_DIR         = _ext('XCP_DIR',         DATA_DIR / 'derivatives' / 'surfaces')
# Preprocessed resting-state dense timeseries used for functional connectivity.
FC_DTSERIES_DIR = _ext('FC_DTSERIES_DIR', DATA_DIR / 'derivatives' / 'rest_timeseries')
# Dynamic covariates file supplying the true year-2 visit age.
AB_G_DYN_FILE   = _ext('AB_G_DYN_FILE',   DATA_DIR / 'covariates' / 'ab_g_dyn.tsv')
# Raw ABCD covariate extracts used to build the extended covariate table.
AB_COVARIATES_DIR = _ext('AB_COVARIATES_DIR', DATA_DIR / 'covariates' / 'raw')

# ── Input data files ──────────────────────────────────────────────────────────
TOPO_FILES = {
    '00A': DATA_DIR / 'network_areas' / '00A_network_areas_cortical.csv',  # baseline (~9-10y)
    '02A': DATA_DIR / 'network_areas' / '02A_network_areas_cortical.csv',  # year 2  (~11-12y)
    '04A': DATA_DIR / 'network_areas' / '04A_network_areas_cortical.csv',  # year 4  (~13-14y)
    '06A': DATA_DIR / 'network_areas' / '06A_network_areas_cortical.csv',  # year 6  (~15-16y)
}

ATLAS_DIR = DATA_DIR / 'atlas_files'
ELA_FILE  = DATA_DIR / 'ela'        / '4_ELA_final.xlsx'
COV_FILE  = DATA_DIR / 'covariates' / '5_covariates_extended.xlsx'
CBCL_FILE = DATA_DIR / 'cbcl'       / 'mh_p_cbcl.tsv'   # ABCD 6.0 release (through year-6)

# Per-timepoint covariate column names.
# age_in_months=True → divide by 12 to get years when loading.
TIMEPOINT_COV = {
    '00A': {'fd': 'rest_mean_FD_baseline', 'site': 'study_site_baseline',
            'age': 'interview_age',       'age_in_months': True},
    # NOTE: covariates file has NO 2-year age column, so 'age' falls back to the
    # baseline 'interview_age' here. df_y2.csv is corrected post-hoc by
    # code/01_data_prep/fix_y2_age.py (true ses-02A visit age, in YEARS, from AB_G_DYN_FILE).
    # RE-APPLY fix_y2_age.py after any phase1 re-run.
    '02A': {'fd': 'rest_mean_FD_2yrFU',   'site': 'study_site_2yrFU',
            'age': 'interview_age',       'age_in_months': True},
    '04A': {'fd': 'rest_mean_FD_4yrFU',   'site': 'study_site_4yrFU',
            'age': 'interview_age_4yrFU', 'age_in_months': False},
    '06A': {'fd': 'rest_mean_FD_6yrFU',   'site': 'study_site_6yrFU',
            'age': 'interview_age_6yrFU', 'age_in_months': False},
}

# ── Analysis constants ────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

NETWORKS = [
    'DMN', 'VIS', 'FP', 'DAN', 'VAN', 'SAL', 'CO',
    'SMD', 'SML', 'AUD', 'Tpole', 'MTL', 'PMN', 'PON', 'SCAN'
]

ELA_COLS = [
    'ELA_caregiver_psych',
    'ELA_ses_neighborhood',
    'ELA_primary_caregiver_support',
    'ELA_secondary_caregiver_support',
    'ELA_family_conflict_youth',
    'ELA_caregiver_substance_sep',
    'ELA_family_anger',
    'ELA_family_aggression',
    'ELA_physical_trauma',
    'ELA_caregiver_supervision',
]

ELA_LABELS = [
    'Caregiver\nPsychopathology',
    'SES/\nNeighborhood',
    'Primary\nCG Support',
    'Secondary\nCG Support',
    'Family\nConflict',
    'CG Substance/\nSeparation',
    'Family\nAnger',
    'Family\nAggression',
    'Physical\nTrauma',
    'CG\nSupervision',
]

ELA_LABELS_SHORT = {
    'ELA_caregiver_psych':              'CG Psych',
    'ELA_ses_neighborhood':             'SES/Neighborhood',
    'ELA_primary_caregiver_support':    'Primary CG Support',
    'ELA_secondary_caregiver_support':  'Secondary CG Support',
    'ELA_family_conflict_youth':        'Family Conflict',
    'ELA_caregiver_substance_sep':      'CG Substance/Sep',
    'ELA_family_anger':                 'Family Anger',
    'ELA_family_aggression':            'Family Aggression',
    'ELA_physical_trauma':              'Physical Trauma',
    'ELA_caregiver_supervision':        'CG Supervision',
}

# Network functional groupings (for visualization and interpretation)
NET_GROUPS = {
    'Transmodal':      ['DMN', 'FP'],
    'Limbic/Salience': ['SAL', 'CO', 'VAN'],
    'Unimodal':        ['SMD', 'SML', 'AUD', 'VIS'],
    'Other/Assoc':     ['DAN', 'MTL', 'PMN', 'PON', 'Tpole', 'SCAN'],
}

NET_GROUP_MAP   = {n: g for g, nets in NET_GROUPS.items() for n in nets}
NET_GROUP_COLOR = {
    'Transmodal':      '#d62728',
    'Limbic/Salience': '#ff7f0e',
    'Unimodal':        '#1f77b4',
    'Other/Assoc':     '#2ca02c',
}

# FDR denominator: 10 ELA × 15 networks
N_TESTS = len(ELA_COLS) * len(NETWORKS)           # 150
BONFERRONI_ALPHA = 0.05 / N_TESTS                 # 0.000333...

# ── ELA PCA higher-order components (legacy — superseded by a priori composites) ─
PC_COLS = ['PC1', 'PC2', 'PC3']

PC_LABELS = {
    'PC1': 'Socioeconomic\nHardship',
    'PC2': 'Caregiver\nDysfunction',
    'PC3': 'Low Parenting\nQuality',
}

PC_LABELS_SHORT = {
    'PC1': 'SES Hardship',
    'PC2': 'CG Dysfunction',
    'PC3': 'Low Parenting',
}

# Variables that need negation so that high = more adversity.
# Applied in both phase0 composite construction and any analysis using raw ELA scores.
ELA_REVERSE_CODED = [
    'ELA_family_anger',
    'ELA_primary_caregiver_support',
    'ELA_secondary_caregiver_support',
]

# ── A priori ELA composites (McLaughlin & Sheridan threat/deprivation/unpred.) ──
# Domain assignments (4 / 4 / 2 split):
#   Threat          — direct harm or ambient hostility
#   Deprivation     — absence of material/parenting resources
#   Unpredictability — caregiver instability producing erratic caregiving
ELA_THREAT_COLS = [
    'ELA_physical_trauma',
    'ELA_family_aggression',
    'ELA_family_conflict_youth',
    'ELA_family_anger',           # negated before compositing
]
ELA_DEPRIVATION_COLS = [
    'ELA_ses_neighborhood',
    'ELA_primary_caregiver_support',   # negated
    'ELA_secondary_caregiver_support', # negated
    'ELA_caregiver_supervision',
]
ELA_UNPRED_COLS = [
    'ELA_caregiver_psych',
    'ELA_caregiver_substance_sep',
]

COMPOSITE_COLS = ['threat_composite', 'deprivation_composite', 'unpredictability_composite']

COMPOSITE_LABELS = {
    'threat_composite':           'Threat',
    'deprivation_composite':      'Deprivation',
    'unpredictability_composite': 'Unpredictability',
}
COMPOSITE_LABELS_SHORT = COMPOSITE_LABELS.copy()

# FDR test count for composite analyses
N_TESTS_COMPOSITES = len(COMPOSITE_COLS) * len(NETWORKS)   # 45

# NIH Toolbox
NIH_TOOLBOX_FILE = DATA_DIR / 'ABCD_NIH_ToolBox' / 'nc_y_nihtb.tsv'
NIH_FLUID_COL    = 'nc_y_nihtb__comp__fluid__agecorr_score'
NIH_CRYST_COL    = 'nc_y_nihtb__comp__cryst__agecorr_score'
NIH_COLS         = [NIH_FLUID_COL, NIH_CRYST_COL]

# CBCL outcome columns
CBCL_OUTCOMES = [
    'cbcl_scr_syn_internal_r',
    'cbcl_scr_syn_external_r',
    'cbcl_scr_syn_totprob_r',
    'cbcl_scr_syn_thought_r',
    'cbcl_scr_syn_attention_r',
    'cbcl_scr_dsm5_depress_r',
    'cbcl_scr_dsm5_anxdisord_r',
    'cbcl_scr_dsm5_adhd_r',
]
CBCL_LABELS = {
    'cbcl_scr_syn_internal_r':   'Internalizing',
    'cbcl_scr_syn_external_r':   'Externalizing',
    'cbcl_scr_syn_totprob_r':    'Total Problems',
    'cbcl_scr_syn_thought_r':    'Thought Problems',
    'cbcl_scr_syn_attention_r':  'Attention',
    'cbcl_scr_dsm5_depress_r':   'Depression (DSM5)',
    'cbcl_scr_dsm5_anxdisord_r': 'Anxiety (DSM5)',
    'cbcl_scr_dsm5_adhd_r':      'ADHD (DSM5)',
}

# Expanded CBCL subscale list for mediation forest plots
CBCL_MEDIATION_OUTCOMES = {
    'cbcl_scr_syn_totprob_r':    'Total Problems',
    'cbcl_scr_syn_internal_r':   'Internalizing',
    'cbcl_scr_syn_external_r':   'Externalizing',
    'cbcl_scr_syn_anxdep_r':     'Anxious/Depressed',
    'cbcl_scr_syn_withdep_r':    'Withdrawn/Depressed',
    'cbcl_scr_syn_somatic_r':    'Somatic Complaints',
    'cbcl_scr_dsm5_depress_r':   'DSM5 Depressive',
    'cbcl_scr_syn_aggressive_r': 'Aggressive Behavior',
    'cbcl_scr_syn_rulebreak_r':  'Rule-Breaking',
    'cbcl_scr_dsm5_conduct_r':   'DSM5 Conduct',
    'cbcl_scr_syn_attention_r':  'Attention Problems',
    'cbcl_scr_dsm5_adhd_r':      'DSM5 ADHD',
    'cbcl_scr_syn_thought_r':    'Thought Problems',
    'cbcl_scr_syn_social_r':     'Social Problems',
}

# ABCD 6.0 mh_p_cbcl.tsv delivers summary scores under new names and BIDS session
# codes. Map the raw-sum columns to the canonical cbcl_scr_*_r names used throughout
# the code so downstream analyses are unchanged. (Verified against mh_p_cbcl.json;
# *_sum is the raw syndrome/DSM-scale sum, the analogue of the old *_r raw score.)
CBCL_SRC_COLUMNS = {
    'mh_p_cbcl_sum':               'cbcl_scr_syn_totprob_r',    # overall Total Problems
    'mh_p_cbcl__synd__int_sum':    'cbcl_scr_syn_internal_r',
    'mh_p_cbcl__synd__ext_sum':    'cbcl_scr_syn_external_r',
    'mh_p_cbcl__synd__anxdep_sum': 'cbcl_scr_syn_anxdep_r',
    'mh_p_cbcl__synd__wthdep_sum': 'cbcl_scr_syn_withdep_r',
    'mh_p_cbcl__synd__som_sum':    'cbcl_scr_syn_somatic_r',
    'mh_p_cbcl__dsm__dep_sum':     'cbcl_scr_dsm5_depress_r',
    'mh_p_cbcl__synd__aggr_sum':   'cbcl_scr_syn_aggressive_r',
    'mh_p_cbcl__synd__rule_sum':   'cbcl_scr_syn_rulebreak_r',
    'mh_p_cbcl__dsm__cond_sum':    'cbcl_scr_dsm5_conduct_r',
    'mh_p_cbcl__synd__attn_sum':   'cbcl_scr_syn_attention_r',
    'mh_p_cbcl__dsm__adhd_sum':    'cbcl_scr_dsm5_adhd_r',
    'mh_p_cbcl__synd__tho_sum':    'cbcl_scr_syn_thought_r',
    'mh_p_cbcl__synd__soc_sum':    'cbcl_scr_syn_social_r',
}

# NIH Toolbox: fluid only available at baseline & year-6; crystallized at all TPs.
# For mediation (ELA-baseline → SCAN-baseline → outcome):
#   CBCL at year-2 (primary); NIH added to df_base as special columns:
NIH_FLUID_Y6_COL    = 'nihtb_fluid_y6'   # year-6 fluid, merged into df_base
NIH_CRYST_Y6_COL    = 'nihtb_cryst_y6'   # year-6 crystallized, merged into df_base
NIH_MEDIATION_COLS  = [NIH_FLUID_Y6_COL, NIH_CRYST_Y6_COL]
NIH_MEDIATION_LABELS = {
    NIH_FLUID_Y6_COL: 'Fluid Cognition (Year-6)',
    NIH_CRYST_Y6_COL: 'Crystallized Cognition (Year-6)',
}

# FDR test counts
N_TESTS_PC     = len(PC_COLS) * len(NETWORKS)                                    # 45
N_TESTS_BEHAV  = len(NETWORKS) * (len(CBCL_OUTCOMES) + len(NIH_MEDIATION_COLS))  # 150


# ── Required input files ──────────────────────────────────────────────────────
# Raw inputs the data-prep stage reads. External derivative directories
# (REPRO_DIR/XCP_DIR/FC_DTSERIES_DIR) are only needed for the HEAVY stages and
# are checked by those scripts themselves.
REQUIRED_INPUT_FILES = {
    'ELA_FILE':         ELA_FILE,
    'COV_FILE':         COV_FILE,
    'CBCL_FILE':        CBCL_FILE,
    'NIH_TOOLBOX_FILE': NIH_TOOLBOX_FILE,
    'AB_G_DYN_FILE':    AB_G_DYN_FILE,
    **{f'TOPO_FILES[{k}]': v for k, v in TOPO_FILES.items()},
}


def check_required_inputs(files=None, strict=True):
    """Verify that required input files exist; return the list of missing ones.

    Call this at the start of a data-prep entry point to fail fast with a clear
    message instead of a deep pandas/read error. With ``strict=True`` a missing
    file raises FileNotFoundError; otherwise the missing list is just returned.
    """
    files = REQUIRED_INPUT_FILES if files is None else files
    missing = [f'{name}: {path}' for name, path in files.items() if not Path(path).exists()]
    if missing and strict:
        raise FileNotFoundError(
            'Missing required input file(s):\n  ' + '\n  '.join(missing) +
            '\n(Set external paths in config.local.sh — see config.local.example.sh.)')
    return missing


# ── Single immutable config object ────────────────────────────────────────────
# Preferred access is `from adtopo.config import cfg` then `cfg.FIG_DIR` etc.,
# rather than importing many individual module globals. `cfg` is read-only so
# configuration cannot be mutated at run time. The module-level names above
# remain defined (cfg draws its values from them), but code should reference
# them through `cfg`.
import types as _types


class _FrozenConfig:
    """Read-only attribute view over a fixed set of config values.

    Picklable (so it can cross joblib/loky process boundaries) via explicit
    __getstate__/__setstate__; __getattr__ guards the backing store name to
    avoid infinite recursion on a not-yet-initialised (unpickled) instance.
    """
    __slots__ = ('_values',)

    def __init__(self, values):
        object.__setattr__(self, '_values', dict(values))

    def __getattr__(self, name):
        # __getattr__ only fires for attributes missing from the instance; guard
        # the backing store so an uninitialised instance raises instead of
        # recursing forever (e.g. during unpickling before _values is set).
        if name == '_values':
            raise AttributeError(name)
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(f'config has no attribute {name!r}')

    def __setattr__(self, name, value):
        raise AttributeError('adtopo config is read-only (cannot set '
                             f'{name!r}); edit adtopo/config.py instead')

    def __getstate__(self):
        return self._values

    def __setstate__(self, state):
        object.__setattr__(self, '_values', state)

    def __dir__(self):
        return sorted(self._values)

    def __repr__(self):
        return f'cfg({len(self._values)} settings)'


# Auto-collect the public API: every module-level constant/function defined
# above (excluding private names, imported modules, and the Path helper).
cfg = _FrozenConfig({
    _k: _v for _k, _v in dict(globals()).items()
    if not _k.startswith('_')
    and _k != 'Path'
    and not isinstance(_v, _types.ModuleType)
    and _k not in ('cfg',)
})
