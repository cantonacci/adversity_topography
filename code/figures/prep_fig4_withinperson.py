"""
Prep for Figure 4c (within-person ΔSCAN → Δcognition). Reuses the exact model
logic of code/05_behavior/within_person/within_person_cognition.py (Result 2, all-waves)
and emits two small tables consumed by fig4_panels.py:

  outputs/tables/fig4c_within_person_coefs.csv
      cryst / fluid all-waves β, cluster-robust SE, 95% CI, p  (for the inset)
  outputs/tables/fig4c_within_person_partial.csv
      per subject-wave partial residuals (resid_dSCAN, resid_dcryst) for the
      added-variable scatter; slope of resid_dcryst~resid_dSCAN = the partial β.

The scatter is the crystallized model; the headline annotated β is the all-waves
LME coefficient (matches the manuscript), and the overlaid line is the OLS
partial-regression fit (added-variable plot).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "code" / "05_behavior" / "within_person" / "derived"
TAB = ROOT / "outputs" / "tables"

z = lambda s: (s - s.mean()) / s.std()


def cluster_robust_lme(df, y_col, x_cols, subject_col, cluster_col, term):
    """MixedLM (random intercept by subject) + family cluster-robust SE.
    Returns (n, n_subjects, beta, se, p) for `term`."""
    needed = [y_col] + x_cols + [subject_col, cluster_col]
    tmp = df[[c for c in needed if c in df.columns]].dropna().copy()
    if tmp["sex"].dtype == object:
        vals = tmp["sex"].unique()
        tmp["sex"] = (tmp["sex"] == vals[0]).astype(float)
    y = tmp[y_col].values.astype(float)
    X = np.column_stack([np.ones(len(tmp))] +
                        [tmp[c].values.astype(float) for c in x_cols])
    groups = tmp[subject_col].values
    cl = tmp[cluster_col].values
    try:
        res = MixedLM(y, X, groups=groups, exog_re=np.ones((len(tmp), 1))).fit(
            reml=True, method="lbfgs", maxiter=300)
        if not res.converged:
            raise RuntimeError("LME did not converge")
        betas = res.fe_params.values
        resid = y - X @ betas
    except Exception:
        betas = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ betas
    n, k = X.shape
    XtXinv = np.linalg.pinv(X.T @ X)
    uniq = np.unique(cl)
    G = len(uniq)
    meat = np.zeros((k, k))
    for c_ in uniq:
        idx = cl == c_
        sc = X[idx].T @ resid[idx]
        meat += np.outer(sc, sc)
    factor = (G / (G - 1)) * (n / (n - k))
    V = XtXinv @ (factor * meat) @ XtXinv
    j = x_cols.index(term) + 1
    b, se = betas[j], np.sqrt(V[j, j])
    p = 2 * stats.t.sf(abs(b / se), df=G - 1)
    tcrit = stats.t.ppf(0.975, df=G - 1)
    return len(tmp), tmp[subject_col].nunique(), b, se, p, b - tcrit * se, b + tcrit * se


def build_all_waves(df, require_cols):
    df = df.sort_values(["subject", "years"])
    cts = df.groupby("subject").size()
    df = df[df["subject"].isin(cts[cts >= 2].index)]
    rows = []
    for subj, g in df.groupby("subject", sort=False):
        g = g.dropna(subset=require_cols)
        if len(g) < 2:
            continue
        t1 = g.iloc[0]
        for _, t2 in g.iloc[1:].iterrows():
            rows.append(dict(
                subject=subj, family=t1["family_id"], sex=t1["sex"], ela=t1["ela"],
                d_sc=t2["scan_prop"] - t1["scan_prop"], base_sc=t1["scan_prop"],
                d_fd=t2["mean_FD"] - t1["mean_FD"], d_years=t2["years"] - t1["years"],
                base_age=t1["age"],
                d_cryst=t2["cog_cryst"] - t1["cog_cryst"], base_cryst=t1["cog_cryst"],
                d_fluid=t2["cog_fluid"] - t1["cog_fluid"], base_fluid=t1["cog_fluid"]))
    return pd.DataFrame(rows)


def partial_resid(df, y_col, x_col, cov_cols):
    """Added-variable residuals: regress y and x on covariates, return residuals
    and the OLS partial slope (resid_y ~ resid_x)."""
    tmp = df[[y_col, x_col] + cov_cols].dropna().copy()
    if tmp["sex"].dtype == object:
        vals = tmp["sex"].unique()
        tmp["sex"] = (tmp["sex"] == vals[0]).astype(float)
    C = np.column_stack([np.ones(len(tmp))] +
                        [tmp[c].values.astype(float) for c in cov_cols])
    def resid(v):
        b = np.linalg.lstsq(C, v, rcond=None)[0]
        return v - C @ b
    rx = resid(tmp[x_col].values.astype(float))
    ry = resid(tmp[y_col].values.astype(float))
    slope = np.linalg.lstsq(np.column_stack([np.ones(len(rx)), rx]), ry, rcond=None)[0][1]
    return rx, ry, slope


def main():
    d = pd.read_csv(DERIVED / "scan_topo_long.csv")
    ela = pd.read_csv(DERIVED / "ela_scores.csv")
    d = d[d["usable"].isin([True, "True", "TRUE"])].copy()
    d = d.merge(ela, on="src_subject_id", how="left")
    d["subject"] = d["src_subject_id"].astype(str)
    d["ela"] = z(d["ela_threat"])

    cov = ["bo", "bs", "d_fd", "d_years", "base_age", "sex"]
    rows = []

    # crystallized
    dc = d[d["scan_prop"].notna() & d["cog_cryst"].notna()].copy()
    awc = build_all_waves(dc, ["scan_prop", "cog_cryst"]).dropna(
        subset=["d_sc", "d_cryst", "base_cryst", "base_sc", "d_fd", "d_years", "base_age"])
    awc["d_sc_z"] = z(awc["d_sc"]); awc["d_o"] = z(awc["d_cryst"])
    awc["bo"] = z(awc["base_cryst"]); awc["bs"] = z(awc["base_sc"])
    n, ns, b, se, p, lo, hi = cluster_robust_lme(
        awc, "d_o", ["d_sc_z"] + cov, "subject", "family", "d_sc_z")
    rows.append(dict(outcome="Crystallized", n=n, n_subj=ns, beta=b, se=se,
                     p=p, ci_lo=lo, ci_hi=hi, sd_draw=awc["d_cryst"].std()))
    rx, ry, slope = partial_resid(awc, "d_o", "d_sc_z", cov)
    pd.DataFrame({"resid_dSCAN": rx, "resid_dcryst": ry}).to_csv(
        TAB / "fig4c_within_person_partial.csv", index=False)
    print(f"cryst  N={n} ({ns} subj)  LME β={b:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4g}"
          f"  | OLS partial slope={slope:+.4f}")

    # fluid
    df_ = d[d["scan_prop"].notna() & d["cog_fluid"].notna()].copy()
    awf = build_all_waves(df_, ["scan_prop", "cog_fluid"]).dropna(
        subset=["d_sc", "d_fluid", "base_fluid", "base_sc", "d_fd", "d_years", "base_age"])
    awf["d_sc_z"] = z(awf["d_sc"]); awf["d_o"] = z(awf["d_fluid"])
    awf["bo"] = z(awf["base_fluid"]); awf["bs"] = z(awf["base_sc"])
    n, ns, b, se, p, lo, hi = cluster_robust_lme(
        awf, "d_o", ["d_sc_z"] + cov, "subject", "family", "d_sc_z")
    rows.append(dict(outcome="Fluid", n=n, n_subj=ns, beta=b, se=se,
                     p=p, ci_lo=lo, ci_hi=hi, sd_draw=awf["d_fluid"].std()))
    print(f"fluid  N={n} ({ns} subj)  LME β={b:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4g}")

    pd.DataFrame(rows).to_csv(TAB / "fig4c_within_person_coefs.csv", index=False)
    print("wrote fig4c_within_person_coefs.csv + fig4c_within_person_partial.csv")


if __name__ == "__main__":
    main()
