"""
Supplement: all-10-ELA-factors multivariate model (Uy #20).
adversity_topography/code/supplement/all10_factors_supplement.py

The main text models the 3 a priori composites; this supplement reports the
model entering all 10 individual ELA factors simultaneously as predictors of
each network's cortical area. The estimates already exist under the canonical
spec (code/02_expansion/multivariate_models.py, tag='individual' →
phase3_individual_results_{wave}.csv, OLS + fixed site + family-cluster-robust
SE). Here we pivot the baseline results into a supplement-ready β matrix with
FDR-significance flags and pull out the SCAN column.

Outputs:
  TAB_DIR/supp_all10factors_beta_matrix_baseline.csv   (10 factors × 15 networks, β; * = q<.05)
  TAB_DIR/supp_all10factors_SCAN_baseline.csv          (SCAN column, full stats)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_CODE = next(a for a in Path(__file__).resolve().parents if (a / 'config.py').exists())
sys.path.insert(0, str(_CODE))
from config import TAB_DIR, NETWORKS, ELA_COLS, ELA_LABELS_SHORT

src = TAB_DIR / 'phase3_individual_results_baseline.csv'
res = pd.read_csv(src)

# β matrix with significance star (q_FDR < .05), predictors × networks
def cell(row):
    b = row['beta']
    star = '*' if (pd.notna(row['q_FDR']) and row['q_FDR'] < 0.05) else ''
    return f'{b:.5f}{star}'

res['cell'] = res.apply(cell, axis=1)
mat = res.pivot(index='predictor', columns='network', values='cell')
mat = mat.reindex(index=[c for c in ELA_COLS if c in mat.index],
                  columns=[n for n in NETWORKS if n in mat.columns])
mat.index = [ELA_LABELS_SHORT.get(i, i) for i in mat.index]
mat.to_csv(TAB_DIR / 'supp_all10factors_beta_matrix_baseline.csv')

# SCAN column, full stats
scan = res[res['network'] == 'SCAN'][['predictor', 'beta', 'se', 't', 'p', 'q_FDR']].copy()
scan['label'] = scan['predictor'].map(lambda p: ELA_LABELS_SHORT.get(p, p))
scan = scan.sort_values('p')
scan.to_csv(TAB_DIR / 'supp_all10factors_SCAN_baseline.csv', index=False)

n_sig_scan = int((scan['q_FDR'] < 0.05).sum())
print(f'Loaded {src.name} ({len(res)} rows).')
print(f'All-10-factor model, SCAN column (sorted by p), baseline:')
print(scan[['label', 'beta', 'p', 'q_FDR']].to_string(index=False))
print(f'\nSCAN: {n_sig_scan}/{len(scan)} individual factors FDR-significant.')
print('Saved: supp_all10factors_beta_matrix_baseline.csv, supp_all10factors_SCAN_baseline.csv')
