"""
Participant-characteristics paragraph: supporting numbers.
  (1) sample descriptors (N, families, sites)
  (2) group-level SCAN topography (size, rank among 15 nets, anatomical seat handled in text)
  (3) sex difference in baseline SCAN proportion
  (4) included (n=4525) vs excluded (rest of ABCD baseline) demographic comparison
Outputs a plain-text summary to outputs/tables/participant_characteristics_summary.txt
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

BASE = Path(__file__).resolve().parents[2]
DAT  = BASE / 'outputs/data_processed/df_base.csv'
COV  = BASE / 'data/covariates/5_covariates_extended.xlsx'
OUT  = BASE / 'outputs/tables/participant_characteristics_summary.txt'

NETWORKS = ['DMN','VIS','FP','DAN','VAN','SAL','CO','SMD','SML','AUD','Tpole','MTL','PMN','PON','SCAN']
L = []
def log(s=''):
    print(s); L.append(str(s))

base = pd.read_csv(DAT)
log("="*70); log("(1) SAMPLE DESCRIPTORS — analytic baseline sample"); log("="*70)
log(f"  N children            : {len(base)}")
log(f"  N unique families     : {base['family_id'].nunique()}")
log(f"  N study sites         : {base['study_site'].nunique()}")
sib = base['family_id'].duplicated(keep=False).sum()
log(f"  Children with a sibling in sample: {sib}")
age_yr = base['interview_age']/12.0
log(f"  Age (yr): mean={age_yr.mean():.2f} SD={age_yr.std():.2f} range[{age_yr.min():.1f},{age_yr.max():.1f}]")
log(f"  sex value_counts: {base['sex'].value_counts().to_dict()}")

log(""); log("="*70); log("(2) GROUP-LEVEL SCAN TOPOGRAPHY"); log("="*70)
prop_cols = [f'prop_{n}' for n in NETWORKS]
med = base[prop_cols].median().sort_values()        # ascending: smallest first
mean= base[prop_cols].mean()
log("  Network sizes (median % of cortex), smallest -> largest:")
for i,(c,v) in enumerate(med.items(),1):
    net = c.replace('prop_','')
    log(f"   {i:2d}. {net:6s} median={v*100:5.2f}%  mean={mean[c]*100:5.2f}%")
scan_rank = list(med.index).index('prop_SCAN')+1
log(f"  SCAN rank by median size: {scan_rank} of {len(NETWORKS)} (1=smallest)")
log(f"  SCAN: mean={base['prop_SCAN'].mean()*100:.2f}%  median={base['prop_SCAN'].median()*100:.2f}%  SD={base['prop_SCAN'].std()*100:.2f}%")
log(f"  SCAN IQR: [{base['prop_SCAN'].quantile(.25)*100:.2f}%, {base['prop_SCAN'].quantile(.75)*100:.2f}%]")
smallest = med.index[0].replace('prop_',''); largest = med.index[-1].replace('prop_','')
log(f"  Smallest network: {smallest} ({med.iloc[0]*100:.2f}%) ; Largest: {largest} ({med.iloc[-1]*100:.2f}%)")

log(""); log("="*70); log("(3) SEX DIFFERENCE IN BASELINE SCAN PROPORTION"); log("="*70)
sx = base[['prop_SCAN','sex']].dropna()
groups = sx.groupby('sex')['prop_SCAN']
levels = list(groups.groups.keys())
g0 = sx.loc[sx['sex']==levels[0],'prop_SCAN']; g1 = sx.loc[sx['sex']==levels[1],'prop_SCAN']
t,p = stats.ttest_ind(g0,g1,equal_var=False)
# Cohen's d (pooled)
nx,ny=len(g0),len(g1); sp=np.sqrt(((nx-1)*g0.var()+(ny-1)*g1.var())/(nx+ny-2)); d=(g0.mean()-g1.mean())/sp
for lv in levels:
    gg=sx.loc[sx['sex']==lv,'prop_SCAN']; log(f"  sex={lv}: n={len(gg)} mean={gg.mean()*100:.3f}% SD={gg.std()*100:.3f}%")
log(f"  Welch t={t:.3f}, p={p:.3g}, Cohen's d={d:.3f}  (level0 - level1 = {levels[0]} - {levels[1]})")

# age & FD associations (covariate justification; not featured in text)
for v in ['interview_age','fd']:
    if v in base.columns:
        dd=base[['prop_SCAN',v]].dropna(); r,pp=stats.pearsonr(dd['prop_SCAN'],dd[v])
        log(f"  prop_SCAN ~ {v}: r={r:.3f}, p={pp:.3g}")

log(""); log("="*70); log("(4) INCLUDED vs EXCLUDED (full ABCD baseline cohort)"); log("="*70)
cov = pd.read_excel(COV)
cov['included'] = cov['sub_ID'].isin(set(base['sub_ID']))
log(f"  COV_FILE N={len(cov)}  included={cov['included'].sum()}  excluded={(~cov['included']).sum()}")
log(f"  COV columns: {list(cov.columns)}")

def cramers_v(ct):
    """Bias-corrected Cramér's V (Bergsma, 2013), which removes the upward bias
    of the uncorrected statistic in large / non-square tables."""
    chi2 = stats.chi2_contingency(ct)[0]; n = ct.values.sum(); r, k = ct.shape
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else np.nan

def cat_compare(col):
    if col not in cov.columns: log(f"  [{col}] not in COV"); return
    sub = cov[[col,'included']].dropna()
    ct = pd.crosstab(sub[col], sub['included'])
    chi2,p,dof,_ = stats.chi2_contingency(ct)
    v = cramers_v(ct)
    log(f"\n  -- {col} -- chi2({dof})={chi2:.2f} p={p:.3g} CramersV_bc={v:.3f}  (n={len(sub)})")
    pct = pd.crosstab(sub[col], sub['included'], normalize='columns')*100
    pct.columns=['excluded%','included%']
    log(pct.round(1).to_string())

for c in ['sex','Race','Income','Parent_edu']:
    cat_compare(c)

# age compare if available
agecol = next((c for c in ['interview_age','age','interview_age_baseline'] if c in cov.columns), None)
if agecol:
    sub=cov[[agecol,'included']].dropna()
    a0=sub.loc[~sub['included'],agecol]; a1=sub.loc[sub['included'],agecol]
    t,p=stats.ttest_ind(a0,a1,equal_var=False)
    log(f"\n  -- age ({agecol}) -- incl mean={a1.mean():.2f} excl mean={a0.mean():.2f} t={t:.2f} p={p:.3g}")
else:
    log("\n  (no age column in COV_FILE for incl/excl age test)")

OUT.write_text("\n".join(L))
log(f"\nSaved -> {OUT}")
