"""Figure S1: correlation matrix of the 10 ELA factors + 3 composites.
Factors are shown in composite-aligned orientation (higher = more adversity),
matching how they enter the threat/deprivation/unpredictability composites.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_CODE = next(a for a in Path(__file__).resolve().parents if (a / 'config.py').exists())
sys.path.insert(0, str(_CODE))
from config import BASE_DIR as ROOT
comp = pd.read_csv(ROOT / 'outputs/tables/ela_composites.csv')

# composite-aligned factor columns (_z) grouped by composite, then the composites
THREAT = ['ELA_physical_trauma_z', 'ELA_family_aggression_z', 'ELA_family_conflict_youth_z', 'ELA_family_anger_z']
DEPRIV = ['ELA_ses_neighborhood_z', 'ELA_primary_caregiver_support_z', 'ELA_secondary_caregiver_support_z', 'ELA_caregiver_supervision_z']
UNPRED = ['ELA_caregiver_psych_z', 'ELA_caregiver_substance_sep_z']
COMPS  = ['threat_composite', 'deprivation_composite', 'unpredictability_composite']
cols = THREAT + DEPRIV + UNPRED + COMPS

labels = {
 'ELA_physical_trauma_z': 'Physical trauma', 'ELA_family_aggression_z': 'Family aggression',
 'ELA_family_conflict_youth_z': 'Family conflict', 'ELA_family_anger_z': 'Family anger †',
 'ELA_ses_neighborhood_z': 'SES / neighborhood', 'ELA_primary_caregiver_support_z': 'Low primary caregiver support †',
 'ELA_secondary_caregiver_support_z': 'Low secondary caregiver support †', 'ELA_caregiver_supervision_z': 'Caregiver supervision',
 'ELA_caregiver_psych_z': 'Caregiver psychopathology', 'ELA_caregiver_substance_sep_z': 'Caregiver substance use / separation',
 'threat_composite': 'THREAT (composite)', 'deprivation_composite': 'DEPRIVATION (composite)',
 'unpredictability_composite': 'UNPREDICTABILITY (composite)',
}
R = comp[cols].corr()

# directionality check: each factor vs its composite
print("Directionality check (factor r with own composite):")
for grp, cc in [(THREAT, 'threat_composite'), (DEPRIV, 'deprivation_composite'), (UNPRED, 'unpredictability_composite')]:
    for f in grp:
        print(f"  {labels[f]:40s} r(with {cc.split('_')[0]}) = {R.loc[f, cc]:+.3f}")

disp = [labels[c] for c in cols]
n = len(cols)
fig, ax = plt.subplots(figsize=(9.2, 8.2))
im = ax.imshow(R.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(disp, rotation=45, ha='right', fontsize=7.5)
ax.set_yticklabels(disp, fontsize=7.5)
# bold the composite tick labels
for i, c in enumerate(cols):
    if c in COMPS:
        ax.get_xticklabels()[i].set_fontweight('bold')
        ax.get_yticklabels()[i].set_fontweight('bold')
# annotate
for i in range(n):
    for j in range(n):
        v = R.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                fontsize=6.0, color='white' if abs(v) > 0.55 else 'black')
# separators between composite blocks and before the composites
for pos in [3.5, 7.5, 9.5]:
    ax.axhline(pos, color='k', lw=1.1); ax.axvline(pos, color='k', lw=1.1)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Pearson r', fontsize=9)
ax.set_title('Intercorrelations among adversity factors and composites', fontsize=11, pad=10)
plt.tight_layout()
outdir = ROOT / 'outputs/figures/supplement'; outdir.mkdir(parents=True, exist_ok=True)
for ext in ['png', 'pdf']:
    fig.savefig(outdir / f'figS1_ela_correlation.{ext}', dpi=300, bbox_inches='tight')
print("Saved:", outdir / 'figS1_ela_correlation.png')
