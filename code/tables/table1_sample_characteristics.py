"""
Table 1 sample-characteristics computation for the ELA-SCAN topography paper.
  - Section 1: baseline analytic-sample Table 1 (full clinical-style)
  - Section 2: longitudinal retention table (N + age + key measures per wave)
  - Section 3: included-vs-excluded generalizability comparison (supplement)
Outputs plain text + a tidy CSV of Table 1 rows for drafting.
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

BASE = Path(__file__).resolve().parents[2]
from adtopo.stats_utils import cramers_v
from adtopo.logging_utils import get_logger
_log = get_logger('table1_sample_characteristics')
DP   = BASE / 'outputs/data_processed'
COV  = BASE / 'data/covariates/5_covariates_extended.xlsx'
OUT  = BASE / 'outputs/tables/table1_sample_characteristics.txt'
OUTC = BASE / 'outputs/tables/table1_baseline_rows.csv'
TR_S = 0.8  # ABCD rsfMRI TR in seconds

L=[]; rows=[]
def log(s=''):
    _log.info(str(s)); L.append(str(s))
def row(label, value, group=''):
    rows.append({'group':group,'label':label,'value':value})

def msd(x, mult=1, dec=2):
    x=pd.Series(x).dropna()*mult
    return f"{x.mean():.{dec}f} ({x.std():.{dec}f})", len(x)
def msd_range(x, mult=1, dec=2):
    x=pd.Series(x).dropna()*mult
    return f"{x.mean():.{dec}f} ({x.std():.{dec}f}); range {x.min():.{dec}f}–{x.max():.{dec}f}", len(x)
def npct(count, denom):
    return f"{count} ({100*count/denom:.1f}%)"

def cat_table(col, order=None, mapper=None, label=None, group="Demographics"):
    s = base[col]
    if mapper is not None: s = s.map(mapper)
    vc = s.value_counts(dropna=True)
    miss = s.isna().sum()
    keys = order if order else list(vc.index)
    log(f"  {label or col}:")
    for k in keys:
        c = int(vc.get(k,0)); row(f"{label or col}: {k}, n (%)",npct(c,N),group)
        log(f"    {str(k):26s}: {npct(c,N)}")
    if miss>0:
        row(f"{label or col}: missing, n (%)",npct(int(miss),N),group)
        log(f"    {'missing':26s}: {npct(int(miss),N)}")

INC_MAP = {**{i:'<$50,000' for i in range(1,7)}, 7:'$50,000–$99,999',8:'$50,000–$99,999',
           9:'≥$100,000',10:'≥$100,000'}

def edu_map(v):
    if pd.isna(v): return np.nan
    v=int(v)
    if v<=12: return '<HS diploma'
    if v in (13,14): return 'HS diploma / GED'
    if v in (15,16,17): return 'Some college / associate'
    if v==18: return "Bachelor's degree"
    if v in (19,20,21): return 'Postgraduate degree'
    return np.nan

def cat_compare(col,mapper=None,label=None,test_raw=True):
    """Included-vs-excluded comparison (Table S3).
    When a mapper is supplied the percentage columns summarize the collapsed
    categories (e.g. income >= $100k, education >= bachelor's), but the
    chi-square and bias-corrected Cramer's V test the FULL raw ABCD category
    distribution (omnibus test: income 10 levels -> df=9, education 21 levels
    -> df=20), unless test_raw=False. Complete-case is defined by the summary
    variable, so n and the missing set match the reported percentages.
    """
    if col not in cov.columns: log(f"  [{col}] not in COV"); return
    s=cov[[col,'included']].dropna().copy()
    if mapper is not None:
        s['_summary']=s[col].map(mapper)
        s=s.dropna(subset=['_summary'])          # complete-case on the summary variable
    else:
        s['_summary']=s[col]
    test_col = col if (mapper is not None and test_raw) else '_summary'
    ct=pd.crosstab(s[test_col],s['included'])
    chi2,p,dof,_=stats.chi2_contingency(ct); v=cramers_v(ct)
    log(f"\n  -- {label or col} -- chi2({dof})={chi2:.2f} p={p:.3g} CramersV_bc={v:.3f}  (n={len(s)})")
    pct=(pd.crosstab(s['_summary'],s['included'],normalize='columns')*100).round(1)
    pct.columns=['excluded%','included%']; log(pct.to_string())

def main():
    global base, N, cov
    base = pd.read_csv(DP/'df_base.csv')
    N = len(base)

    log("="*72); log("SECTION 1 — BASELINE ANALYTIC-SAMPLE TABLE 1 (N={})".format(N)); log("="*72)

    # --- Sample / design ---
    log("\n[Sample / design]")
    row("N children","{}".format(N),"Sample")
    row("N unique families","{}".format(base['family_id'].nunique()),"Sample")
    row("N children with a sibling in sample","{}".format(int(base['family_id'].duplicated(keep=False).sum())),"Sample")
    row("N study sites","{}".format(base['study_site'].nunique()),"Sample")
    for r in rows[-4:]: log(f"  {r['label']:42s}: {r['value']}")

    # --- Demographics ---
    log("\n[Demographics]")
    age_v,age_n = msd_range(base['interview_age'],1,2)   # already in YEARS
    row("Age, years, mean (SD)",age_v,"Demographics"); log(f"  Age (yr): {age_v}  [n={age_n}]")
    sexvc = base['sex'].value_counts()
    for lv in ['F','M']:
        if lv in sexvc.index:
            row(f"Sex = {lv}, n (%)",npct(int(sexvc[lv]),N),"Demographics")
            log(f"  Sex={lv}: {npct(int(sexvc[lv]),N)}")

    cat_table('Race', order=['White','Black/AA','Hispanic','Asian','Other/Multiracial'], label='Race/ethnicity')

    cat_table('Income', order=['<$50,000','$50,000–$99,999','≥$100,000'], mapper=INC_MAP, label='Household income')
    cat_table('Parent_edu', order=['<HS diploma','HS diploma / GED','Some college / associate',"Bachelor's degree",'Postgraduate degree'], mapper=edu_map, label='Highest parental education')

    # --- Adversity exposures ---
    log("\n[Early-life adversity exposures]")
    for comp,lab in [('threat_composite','Threat composite'),('deprivation_composite','Deprivation composite'),('unpredictability_composite','Unpredictability composite')]:
        v,n = msd_range(base[comp],1,2)
        row(f"{lab}, mean (SD)",v,"Adversity"); log(f"  {lab}: {v}  [n={n}]")
    thr = base['threat_composite']
    hi = int((thr>=(thr.mean()+thr.std())).sum()); lo=int((thr<=(thr.mean()-thr.std())).sum())
    row("High-threat (≥+1 SD), n",f"{hi}","Adversity"); row("Low-threat (≤−1 SD), n",f"{lo}","Adversity")
    log(f"  High-threat (≥+1SD) n={hi} ; Low-threat (≤−1SD) n={lo}")

    # --- Imaging / motion QC ---
    log("\n[Imaging & motion QC, baseline]")
    fd_v,_ = msd(base['fd'],1,3); row("Mean framewise displacement, mm, mean (SD)",fd_v,"Imaging QC"); log(f"  Mean FD (mm): {fd_v}")
    fr = base['rest_total_frames_post_scrubbing_baseline']
    fr_v,_ = msd(fr,1,0); row("Retained rest volumes post-scrubbing, mean (SD)",fr_v,"Imaging QC"); log(f"  Retained frames: {fr_v}")
    minv,_ = msd(fr*TR_S/60.0,1,1); row("Retained rest data, minutes, mean (SD)",minv,"Imaging QC"); log(f"  Retained minutes: {minv}")
    mvc = base['scanner_manufacturer_baseline'].value_counts()
    name_map={'SIEMENS':'Siemens','GE MEDICAL SYSTEMS':'GE','Philips Medical Systems':'Philips'}
    log("  Scanner manufacturer:")
    for k in ['SIEMENS','GE MEDICAL SYSTEMS','Philips Medical Systems']:
        c=int(mvc.get(k,0)); row(f"Scanner: {name_map[k]}, n (%)",npct(c,N),"Imaging QC"); log(f"    {name_map[k]:10s}: {npct(c,N)}")

    # --- Primary topographic measure ---
    log("\n[Primary topographic measure]")
    scan=base['prop_SCAN']*100
    row("SCAN territory, % cortex, mean (SD)",f"{scan.mean():.2f} ({scan.std():.2f})","SCAN")
    row("SCAN territory, % cortex, median [IQR]",f"{scan.median():.2f} [{scan.quantile(.25):.2f}–{scan.quantile(.75):.2f}]","SCAN")
    log(f"  SCAN % cortex: mean(SD)={scan.mean():.2f}({scan.std():.2f})  median[IQR]={scan.median():.2f}[{scan.quantile(.25):.2f}-{scan.quantile(.75):.2f}]")

    # --- Cognitive & behavioral outcomes ---
    log("\n[Cognitive & behavioral outcomes]")
    cog = [('nc_y_nihtb__comp__cryst__agecorr_score','NIH-TB Crystallized (age-corr), baseline'),
           ('nc_y_nihtb__comp__fluid__agecorr_score','NIH-TB Fluid (age-corr), baseline'),
           ('nihtb_cryst_y6','NIH-TB Crystallized (age-corr), year 6'),
           ('nihtb_fluid_y6','NIH-TB Fluid (age-corr), year 6')]
    for c,lab in cog:
        if c in base.columns:
            v,n=msd(base[c],1,2); row(f"{lab}, mean (SD)",v,"Outcomes"); row(f"{lab}, N","{}".format(n),"Outcomes")
            log(f"  {lab}: {v}  [n={n}]")
    cbcl=[('cbcl_scr_syn_totprob_r','CBCL Total Problems (raw t? r)'),
          ('cbcl_scr_syn_attention_r','CBCL Attention Problems'),
          ('cbcl_scr_dsm5_adhd_r','CBCL DSM-5 ADHD'),
          ('cbcl_scr_syn_thought_r','CBCL Thought Problems')]
    for c,lab in cbcl:
        if c in base.columns:
            v,n=msd(base[c],1,2); row(f"{lab}, mean (SD)",v,"Outcomes")
            log(f"  {lab}: {v}  [n={n}]")

    # ---------------- SECTION 2: retention across waves ----------------
    log("\n"+"="*72); log("SECTION 2 — LONGITUDINAL RETENTION (per wave)"); log("="*72)
    waves=[('Baseline','df_base.csv','interview_age'),
           ('Year 2','df_y2.csv','interview_age'),
           ('Year 4','df_y4.csv','interview_age'),
           ('Year 6','df_y6.csv','interview_age')]
    log(f"  {'Wave':10s} {'N':>6s} {'Age yr mean(SD)':>18s} {'%F':>6s}")
    for wname,fn,agecol in waves:
        d=pd.read_csv(DP/fn)
        a=d[agecol].dropna() if agecol in d.columns else pd.Series(dtype=float)
        pf = 100*(d['sex']=='F').mean() if 'sex' in d.columns else float('nan')
        astr = f"{a.mean():.2f} ({a.std():.2f})" if len(a) else "NA"
        log(f"  {wname:10s} {len(d):6d} {astr:>18s} {pf:6.1f}")
        row(f"{wname}: N",str(len(d)),"Retention")
        if len(a): row(f"{wname}: Age yr mean (SD)",astr,"Retention")

    # ---------------- SECTION 3: included vs excluded (supplement) ----------------
    log("\n"+"="*72); log("SECTION 3 — INCLUDED vs EXCLUDED (supplement)"); log("="*72)
    cov=pd.read_excel(COV)
    cov['included']=cov['sub_ID'].isin(set(base['sub_ID']))
    log(f"  Full baseline cohort N={len(cov)}  included={int(cov['included'].sum())}  excluded={int((~cov['included']).sum())}")
    cat_compare('sex')
    cat_compare('Race')
    cat_compare('Income',mapper=INC_MAP,label='Household income (omnibus chi2 on 10 raw levels; % >= $100k)')
    cat_compare('Parent_edu',mapper=edu_map,label='Parental education (omnibus chi2 on 21 raw levels; % >= bachelor\'s)')

    pd.DataFrame(rows).to_csv(OUTC,index=False)
    OUT.write_text("\n".join(L))
    log(f"\nSaved -> {OUT}")
    log(f"Saved -> {OUTC}")

if __name__ == '__main__':
    main()
