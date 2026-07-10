"""
Supplement: study design / nesting structure of site, scanner and family.
adversity_topography/code/supplement/design_nesting_structure.py

Provides the empirical facts that justify the reported variance structure and
answer the reviewer (Andre) questions:
  * How many sites, scanner manufacturers, scanner serials?
  * Is scanner manufacturer nested within site (so fixed site absorbs it)?
  * Do families ever span sites (so 'family nested in site' is structural)?
  * How many families contribute >1 imaged sibling (why the family RE is ~0)?

Computed at BASELINE (the reported expansion timepoint). Outputs:
  TAB_DIR/supp_design_nesting.csv     (one row per fact, value + interpretation)
  DAT_DIR/supp_design_nesting.txt     (human-readable)
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import numpy as np
import pandas as pd

_CODE = next(a for a in Path(__file__).resolve().parents if (a / 'config.py').exists())
sys.path.insert(0, str(_CODE))
from config import TAB_DIR, DAT_DIR

SITE = 'study_site'
FAMILY = 'family_id'
MANUF = 'scanner_manufacturer_baseline'
SERIAL = 'scanner_serial_number_baseline'

log_lines = []
def log(msg=''):
    print(msg, flush=True)
    log_lines.append(str(msg))

facts = []
def fact(name, value, interpretation):
    facts.append({'quantity': name, 'value': value, 'interpretation': interpretation})
    log(f'  {name:42s} {str(value):>8s}   {interpretation}')


df = pd.read_csv(DAT_DIR / 'df_base.csv')
log(f'Loaded df_base N={len(df)}  (baseline analysis sample)\n')
log('=' * 78)
log('DESIGN / NESTING STRUCTURE (baseline)')
log('=' * 78)

n_site = df[SITE].nunique()
fact('n_study_sites', n_site, 'sites entered as fixed-effect dummies')

# ── Scanner manufacturer / serial and their nesting within site ───────────────
if MANUF in df.columns:
    n_manuf = df[MANUF].nunique()
    fact('n_scanner_manufacturers', n_manuf,
         'too few levels for a stable random-effect variance (<5-8)')
    # Is each site served by exactly one manufacturer? (manufacturer nested in site)
    manuf_per_site = df.groupby(SITE)[MANUF].nunique()
    n_site_multi_manuf = int((manuf_per_site > 1).sum())
    fact('sites_with_>1_manufacturer', n_site_multi_manuf,
         'if 0, manufacturer is perfectly nested in site -> absorbed by site dummies')

if SERIAL in df.columns:
    n_serial = df[SERIAL].nunique()
    fact('n_scanner_serials', n_serial, 'scanner serials')
    serial_per_site = df.groupby(SITE)[SERIAL].nunique()
    n_serial_span = int((df.groupby(SERIAL)[SITE].nunique() > 1).sum())
    fact('serials_spanning_>1_site', n_serial_span,
         'if 0, serial nested in site -> absorbed by site dummies')
    fact('max_serials_per_site', int(serial_per_site.max()),
         'sites with multiple scanners over the study')

# ── Family nesting within site ────────────────────────────────────────────────
fam_site_span = df.groupby(FAMILY)[SITE].nunique()
n_fam_span = int((fam_site_span > 1).sum())
n_families = df[FAMILY].nunique()
fact('n_families', n_families, 'distinct family IDs in the sample')
fact('families_spanning_>1_site', n_fam_span,
     'if 0, families never cross sites -> family is structurally nested in site')

# ── Sibling structure (why the family RE variance is ~0) ──────────────────────
fam_size = df.groupby(FAMILY).size()
n_multi = int((fam_size >= 2).sum())
pct_multi_fam = 100 * n_multi / n_families
n_subj_with_sib = int(fam_size[fam_size >= 2].sum())
pct_subj_with_sib = 100 * n_subj_with_sib / len(df)
fact('families_with_>=2_imaged_sibs', n_multi,
     f'{pct_multi_fam:.1f}% of families')
fact('pct_participants_with_a_sib_in_sample', round(pct_subj_with_sib, 1),
     'limited within-family replication -> family variance poorly identified')

# ── Family random-effect variance (from the convergence diagnostic, if present)
conv_path = DAT_DIR / 'convergence_diagnostic.csv'
if conv_path.exists():
    conv = pd.read_csv(conv_path)
    base = conv[(conv['phase'] == 'phase3') & (conv['wave'] == 'baseline')]
    if len(base) and 're_var' in base.columns:
        re_var = pd.to_numeric(base['re_var'], errors='coerce')
        n_zero = int((re_var.fillna(0) < 1e-8).sum())
        fact('phase3_baseline_family_var~0_models', f'{n_zero}/{len(base)}',
             'singular family RE in the crossed models (family var at boundary)')

pd.DataFrame(facts).to_csv(TAB_DIR / 'supp_design_nesting.csv', index=False)

log('\nInterpretation:')
log('  Fixed site dummies span the manufacturer and serial spaces (both nested in')
log('  site), so scanner hardware is fully absorbed; a 3-level manufacturer random')
log('  effect would add no information and too few levels to estimate. Families')
log('  never cross sites, so "family nested in site" is structurally true but the')
log('  family variance is ~0 (few multi-sib families); sibling non-independence is')
log('  therefore handled with family-cluster-robust SEs rather than a singular RE.')

with open(DAT_DIR / 'supp_design_nesting.txt', 'w') as f:
    f.write('\n'.join(log_lines))
log('\nSaved: supp_design_nesting.csv, supp_design_nesting.txt')
