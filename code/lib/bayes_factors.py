"""
Shared BIC-approximation Bayes-factor helpers for the supplement scripts.

Consolidated verbatim from code/supplement/bayes_factors_expansion.py and
code/supplement/bayes_factors_individual.py, which defined these three
functions byte-identically. BF10 = exp((BIC_null - BIC_full) / 2); BF10 > 1
favors an association (H1), BF10 < 1 favors the null (report BF01 = 1/BF10).
Kass & Raftery (1995): >3 positive, >20 strong, >150 very strong.
"""
import numpy as np
import statsmodels.formula.api as smf

SITE = 'study_site'


def _bic(df, outcome, terms):
    """BIC of OLS: outcome ~ terms + C(site). terms is a list (may be empty)."""
    rhs = ' + '.join(terms + [f'C({SITE})']) if terms else f'C({SITE})'
    return smf.ols(f'{outcome} ~ {rhs}', data=df).fit().bic


def bf10_from_bic(bic_null, bic_full):
    return float(np.exp((bic_null - bic_full) / 2.0))


def label_bf(bf10):
    """Kass & Raftery verbal label for the evidence (toward H1 or H0)."""
    bf, direction = (bf10, 'H1') if bf10 >= 1 else (1.0 / bf10, 'H0')
    if bf < 3:    strength = 'not worth mentioning'
    elif bf < 20: strength = 'positive'
    elif bf < 150: strength = 'strong'
    else:         strength = 'very strong'
    return f'{strength} for {direction}'
