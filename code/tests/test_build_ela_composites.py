"""ELA composite construction: make_composite is a mean-then-standardize."""
import numpy as np
import pandas as pd
import build_ela_composites as bec


def test_make_composite_is_standardized():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=['x1', 'x2', 'x3'])
    comp = bec.make_composite(X, ['x1', 'x2', 'x3'])
    assert abs(comp.mean()) < 1e-9
    assert abs(comp.std() - 1.0) < 1e-9          # pandas .std defaults to ddof=1


def test_make_composite_matches_manual():
    X = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [2.0, 0.0, 4.0, 1.0]})
    comp = bec.make_composite(X, ['a', 'b'])
    raw = X[['a', 'b']].mean(axis=1)
    expected = (raw - raw.mean()) / raw.std()
    pd.testing.assert_series_equal(comp, expected, check_names=False)


def test_make_composite_uses_only_named_columns():
    # a decoy column must not influence the composite
    X = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0],
                      'b': [2.0, 0.0, 4.0, 1.0],
                      'decoy': [99.0, -99.0, 5.0, 0.0]})
    comp_ab = bec.make_composite(X, ['a', 'b'])
    comp_ab_only = bec.make_composite(X[['a', 'b']], ['a', 'b'])
    pd.testing.assert_series_equal(comp_ab, comp_ab_only)
