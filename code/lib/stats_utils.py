"""
Shared statistics utilities.

Consolidated verbatim from code/tables/table1_sample_characteristics.py and
code/tables/participant_characteristics.py, which defined cramers_v
byte-identically.
"""
import numpy as np
from scipy import stats


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
