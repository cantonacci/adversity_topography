"""
Shared random-effects model helper for cross-sectional ELA -> outcome associations.
adversity_topography/code/lib/re_models.py

One function, ``fit_spec()``, fits

    outcome ~ predictors + covariates + site

under a chosen variance structure and returns the fixed-effect estimate for a
target predictor in a fixed record schema, so results from several
specifications stack into one table.

It exists to (a) produce the random-effects specification-invariance supplement
requested by reviewers (Andre: why fixed site? nest scanner? nest family in
site?), and (b) after adoption, be the single place the reported cross-sectional
models (expansion, FC, mediation path-a) obtain their site/family variance
structure, so one auditable specification is applied everywhere.

Specifications
--------------
ols_cluster            OLS, site fixed dummies, family-cluster-robust SEs   [REPORTED TARGET]
ols_classical          OLS, site fixed dummies, classical (iid) SEs
re_family              MixedLM, site fixed dummies, random family intercept  [current phase-2 / mediation path-a]
re_site                MixedLM, random site intercept (no site dummies)      [Andre: site as random]
re_crossed             MixedLM, random site VC + random family intercept     [current phase-3 / FC]
re_family_nested_site  MixedLM, random site + random family-within-site      [Andre: family nested in site]
re_scanner_in_site     MixedLM, site fixed dummies + random scanner-serial VC [Andre: scanner make/serial]

Every spec reports the same fields (beta, se, z/t, p, partial_r, n, converged,
variance components), so ``fit_spec`` can be looped over SPECS for one row per
specification.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# All specifications, in the order they should appear in a supplement table.
SPECS = [
    'ols_cluster',
    'ols_classical',
    're_family',
    're_site',
    're_crossed',
    're_family_nested_site',
    're_scanner_in_site',
]

SPEC_LABELS = {
    'ols_cluster':           'OLS, fixed site + family cluster-robust SE',
    'ols_classical':         'OLS, fixed site, classical SE',
    're_family':             'MixedLM, fixed site + random family intercept',
    're_site':               'MixedLM, random site intercept',
    're_crossed':            'MixedLM, crossed random site + random family',
    're_family_nested_site': 'MixedLM, random site + family nested in site',
    're_scanner_in_site':    'MixedLM, fixed site + random scanner-serial + random family',
}


def _robust_fit(md):
    """Fit a MixedLM with a reproducible optimizer cascade.

    Forcing a single optimizer (e.g. method='lbfgs') can throw a
    ``LinAlgError: Singular matrix`` when a variance component sits at its zero
    boundary (the family RE here). The statsmodels default sequence
    (BFGS -> LBFGS -> CG) converges these; Nelder-Mead is a final fallback.
    Returns the first converged fit, else the last successful fit, else raises.
    """
    last = None
    for kw in ({'reml': True},
               {'reml': True, 'method': 'cg', 'maxiter': 2000},
               {'reml': True, 'method': 'nm', 'maxiter': 4000}):
        try:
            res = md.fit(**kw)
            last = res
            if getattr(res, 'converged', False):
                return res
        except Exception:
            continue
    if last is not None:
        return last
    raise RuntimeError('all optimizers failed')


def _covre00(res):
    """Group random-intercept variance (cov_re[0,0]); nan if unavailable."""
    try:
        return float(np.atleast_2d(res.cov_re)[0, 0])
    except Exception:
        return np.nan


def _vcomp(res, i=0):
    """i-th variance component (res.vcomp[i]); nan if unavailable/empty."""
    try:
        v = res.vcomp
        return float(v[i]) if v is not None and len(v) > i else np.nan
    except Exception:
        return np.nan


def _partial_r(t, n, k):
    """Partial correlation of the target predictor from its t-statistic.

    k = number of fixed-effect parameters (including intercept). For mixed
    models the residual df is approximate; treated as n - k throughout for a
    common, comparable effect-size metric.
    """
    df_resid = max(n - k, 1)
    return float(np.sign(t) * np.sqrt(t ** 2 / (t ** 2 + df_resid)))


def _blank(spec, outcome, target, n=np.nan, note=''):
    return {
        'spec': spec, 'spec_label': SPEC_LABELS.get(spec, spec),
        'outcome': outcome, 'target': target,
        'beta': np.nan, 'se': np.nan, 'stat': np.nan, 'p': np.nan,
        'partial_r': np.nan, 'n': n, 'converged': False,
        'var_components': '', 'note': note,
    }


def fit_spec(df, outcome, target, predictors, covariates, spec,
             site_col='study_site', family_col='family_id',
             scanner_col='scanner_serial_number_baseline',
             maxiter=500):
    """Fit ``outcome ~ predictors + covariates (+ site)`` under ``spec``.

    Parameters
    ----------
    df          : DataFrame holding all needed columns.
    outcome     : outcome column name (e.g. 'prop_SCAN').
    target      : the predictor whose estimate is returned (must be in predictors).
    predictors  : list of fixed-effect predictors of interest (e.g. composites).
    covariates  : list of nuisance covariates (e.g. ['interview_age','sex_num','fd']).
    spec        : one of SPECS.
    site_col, family_col, scanner_col : grouping columns.

    Returns a single-record dict (see ``_blank`` for the schema).
    """
    if spec not in SPECS:
        raise ValueError(f'unknown spec: {spec!r}; choose from {SPECS}')
    if target not in predictors:
        raise ValueError(f'target {target!r} must be one of predictors {predictors}')

    fixed = list(predictors) + list(covariates)
    needed = [outcome] + fixed + [site_col, family_col]
    if spec == 're_scanner_in_site':
        needed.append(scanner_col)
    needed = [c for c in dict.fromkeys(needed) if c in df.columns]

    missing = [c for c in ([outcome, target] + list(covariates)) if c not in df.columns]
    if missing:
        return _blank(spec, outcome, target, note=f'missing columns: {missing}')

    tmp = df[needed].dropna().copy()
    tmp[site_col] = tmp[site_col].astype(str)
    if scanner_col in tmp.columns:
        tmp[scanner_col] = tmp[scanner_col].astype(str)
    n = len(tmp)
    if n < 50:
        return _blank(spec, outcome, target, n=n, note='n<50')

    rhs_fixed = ' + '.join(fixed)
    formula_site = f'{outcome} ~ {rhs_fixed} + C({site_col})'   # site as fixed dummies
    formula_nosite = f'{outcome} ~ {rhs_fixed}'                 # site absorbed by RE

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            # ---- OLS specifications -----------------------------------------
            if spec in ('ols_cluster', 'ols_classical'):
                mod = smf.ols(formula_site, data=tmp)
                if spec == 'ols_cluster':
                    res = mod.fit(cov_type='cluster',
                                  cov_kwds={'groups': tmp[family_col].values})
                    stat_kind = 'z'
                else:
                    res = mod.fit()
                    stat_kind = 't'
                k = int(res.df_model) + 1
                beta = float(res.params[target]); se = float(res.bse[target])
                stat = float(res.tvalues[target]); p = float(res.pvalues[target])
                n_fam = tmp[family_col].nunique()
                return {**_blank(spec, outcome, target, n=n),
                        'beta': beta, 'se': se, 'stat': stat, 'p': p,
                        'partial_r': _partial_r(stat, n, k), 'converged': True,
                        'var_components': f'{stat_kind}; n_families={n_fam}',
                        'note': ''}

            # ---- Mixed-effects specifications --------------------------------
            if spec == 're_family':
                res = _robust_fit(smf.mixedlm(formula_site, data=tmp,
                                              groups=tmp[family_col]))
                vc = f'family_var={_covre00(res):.3g}'

            elif spec == 're_site':
                res = _robust_fit(smf.mixedlm(formula_nosite, data=tmp,
                                              groups=tmp[site_col]))
                vc = f'site_var={_covre00(res):.3g}'

            elif spec == 're_crossed':
                res = _robust_fit(smf.mixedlm(formula_nosite, data=tmp, groups=tmp[family_col],
                                              vc_formula={'site': f'0 + C({site_col})'}))
                vc = f'family_var={_covre00(res):.3g}; site_var={_vcomp(res):.3g}'

            elif spec == 're_family_nested_site':
                # (1 | site) + (1 | site:family): site is the group, family a VC within it.
                res = _robust_fit(smf.mixedlm(formula_nosite, data=tmp, groups=tmp[site_col],
                                              vc_formula={'family': f'0 + C({family_col})'}))
                vc = f'site_var={_covre00(res):.3g}; family_within_site_var={_vcomp(res):.3g}'

            elif spec == 're_scanner_in_site':
                # Fixed site dummies + random scanner-serial VC + random family:
                # demonstrates scanner variance on top of fixed site is ~0.
                if scanner_col not in tmp.columns:
                    return _blank(spec, outcome, target, n=n,
                                  note=f'missing scanner column {scanner_col}')
                res = _robust_fit(smf.mixedlm(formula_site, data=tmp, groups=tmp[family_col],
                                              vc_formula={'scanner': f'0 + C({scanner_col})'}))
                vc = f'family_var={_covre00(res):.3g}; scanner_serial_var={_vcomp(res):.3g}'

            # Use the full params/bse/pvalues Series (label-indexed by predictor
            # name), matching the proven pipeline extraction. res.bse_fe can raise
            # on the singular-RE Hessian; res.bse (full) is robust.
            k = len(res.fe_params)
            beta = float(res.params[target]); se = float(res.bse[target])
            stat = float(res.tvalues[target]); p = float(res.pvalues[target])
            return {**_blank(spec, outcome, target, n=n),
                    'beta': beta, 'se': se, 'stat': stat, 'p': p,
                    'partial_r': _partial_r(stat, n, k),
                    'converged': bool(getattr(res, 'converged', True)),
                    'var_components': vc, 'note': ''}

        except Exception as exc:                       # pragma: no cover - diagnostics
            return _blank(spec, outcome, target, n=n, note=f'{type(exc).__name__}: {exc}')


def fit_all_specs(df, outcome, target, predictors, covariates, **kw):
    """Run every spec in SPECS and return a tidy DataFrame (one row per spec)."""
    rows = [fit_spec(df, outcome, target, predictors, covariates, spec, **kw)
            for spec in SPECS]
    return pd.DataFrame(rows)


# ── Canonical reported specification ──────────────────────────────────────────
# The cross-sectional models reported in the manuscript (SCAN expansion, FC,
# mediation path-a) all use this one structure so a single specification can be
# stated in Methods: OLS with study site as fixed-effect dummies and
# family-cluster-robust (sandwich) standard errors. Site fixed effects
# exhaustively absorb scanner make/model/serial (all nested within site);
# family-clustered SEs account for siblings without a singular random effect.
CANONICAL_SPEC = 'ols_cluster'


def fit_ols_cluster_table(df, outcome, predictors, covariates,
                          site_col='study_site', family_col='family_id'):
    """Fit the canonical reported model once and return every predictor's estimate.

        outcome ~ predictors + covariates + C(site)      [OLS, family cluster-robust SE]

    Returns (table, meta) where ``table`` is a DataFrame with columns
    [predictor, beta, se, z, p, partial_r] (one row per entry in ``predictors``)
    and ``meta`` is a dict {n, n_families, n_params, converged, method}.
    On failure returns (empty DataFrame, meta with converged=False).
    """
    fixed = list(predictors) + list(covariates)
    needed = [c for c in dict.fromkeys([outcome] + fixed + [site_col, family_col])
              if c in df.columns]
    tmp = df[needed].dropna().copy()
    tmp[site_col] = tmp[site_col].astype(str)
    n = len(tmp)
    meta = {'n': n, 'n_families': int(tmp[family_col].nunique()) if n else 0,
            'n_params': 0, 'converged': False, 'method': CANONICAL_SPEC}
    if n < 50:
        return pd.DataFrame(columns=['predictor', 'beta', 'se', 'z', 'p', 'partial_r']), meta

    formula = f'{outcome} ~ ' + ' + '.join(fixed) + f' + C({site_col})'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = smf.ols(formula, data=tmp).fit(
            cov_type='cluster', cov_kwds={'groups': tmp[family_col].values})
    k = int(res.df_model) + 1
    meta.update({'n_params': k, 'converged': True})
    rows = []
    for p_ in predictors:
        if p_ not in res.params:
            continue
        z = float(res.tvalues[p_])
        rows.append({'predictor': p_, 'beta': float(res.params[p_]),
                     'se': float(res.bse[p_]), 'z': z, 'p': float(res.pvalues[p_]),
                     'partial_r': _partial_r(z, n, k)})
    return pd.DataFrame(rows), meta
