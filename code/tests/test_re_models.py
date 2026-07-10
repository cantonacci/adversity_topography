"""fit_ols_cluster_table: schema, explicit method arg, and ModelFitter parity."""
import numpy as np
import pandas as pd
import adtopo.re_models as rm


def _synth(n=300, seed=1):
    rng = np.random.default_rng(seed)
    fam = np.repeat(np.arange(n // 2), 2)[:n]        # sibling pairs
    return pd.DataFrame({
        'y':             0.5 * rng.normal(size=n),
        'x':             rng.normal(size=n),
        'interview_age': rng.normal(12, 1, n),
        'sex_num':       rng.integers(0, 2, n),
        'fd':            rng.random(n) * 0.3,
        'study_site':    rng.integers(0, 4, n),
        'family_id':     fam,
    })


def test_schema_and_default_method():
    df = _synth()
    tbl, meta = rm.fit_ols_cluster_table(df, 'y', ['x'], ['interview_age', 'sex_num', 'fd'])
    assert list(tbl.columns) == ['predictor', 'beta', 'se', 'z', 'p', 'partial_r']
    assert len(tbl) == 1 and tbl.iloc[0]['predictor'] == 'x'
    assert meta['converged'] and meta['n'] == len(df)
    assert meta['method'] == 'ols_cluster'          # default label


def test_method_is_an_argument_not_a_global():
    df = _synth()
    _, meta = rm.fit_ols_cluster_table(df, 'y', ['x'], ['interview_age', 'sex_num', 'fd'],
                                       method='custom_label')
    assert meta['method'] == 'custom_label'


def test_modelfitter_matches_functional_api():
    df = _synth()
    tbl_fn, meta_fn = rm.fit_ols_cluster_table(df, 'y', ['x'],
                                               ['interview_age', 'sex_num', 'fd'])
    mf = rm.ModelFitter(df, 'y', 'x', ['x'], ['interview_age', 'sex_num', 'fd'])
    tbl_cls, meta_cls = mf.canonical_table()
    pd.testing.assert_frame_equal(tbl_fn, tbl_cls)
    assert meta_fn == meta_cls
