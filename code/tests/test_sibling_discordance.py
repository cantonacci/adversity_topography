"""Frisch-Waugh-Lovell equivalence underlying sibling_discordance.py.

The canonical within-family estimator demeans each variable by its family mean
and fits OLS without an intercept; sibling_discordance.py asserts (in a comment)
that this equals an explicit family-fixed-effects (LSDV) fit. That equivalence
is a theorem (FWL) for the point estimate; this test confirms it numerically so
the claim is backed by a check rather than an unverified note.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _sibling_data(seed=3, n_fam=150):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_fam):
        fam_effect = rng.normal()                 # shared family confound
        for _ in range(int(rng.integers(2, 4))):  # 2-3 siblings
            x = rng.normal()
            y = fam_effect + 0.4 * x + rng.normal() * 0.5
            rows.append((f, x, y))
    return pd.DataFrame(rows, columns=['fam', 'x', 'y'])


def test_within_demeaned_equals_family_fixed_effects():
    d = _sibling_data()
    # within-family demeaned OLS (no intercept) — the canonical estimator
    d['x_dm'] = d['x'] - d.groupby('fam')['x'].transform('mean')
    d['y_dm'] = d['y'] - d.groupby('fam')['y'].transform('mean')
    b_within = sm.OLS(d['y_dm'].values, d['x_dm'].values).fit().params[0]

    # explicit family dummies (LSDV)
    dummies = pd.get_dummies(d['fam'], drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([d[['x']].reset_index(drop=True),
                                   dummies.reset_index(drop=True)], axis=1))
    b_lsdv = sm.OLS(d['y'].values, X.values).fit().params[1]  # const, x, dummies...

    assert abs(b_within - b_lsdv) < 1e-8
