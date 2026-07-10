"""Accuracy + idempotency of the Year-2 age correction (fix_y2_age.py)."""
import pandas as pd
import fix_y2_age


def _df_y2():
    # df_y2 before correction: interview_age still holds the placeholder baseline age.
    return pd.DataFrame({
        'sub_ID':        ['a', 'b', 'c', 'd'],
        'interview_age': [9.9, 10.0, 9.8, 10.1],
        'prop_SCAN':     [0.020, 0.030, 0.025, 0.028],
    })


def _y2age():
    # true ses-02A visit ages for a, b, c; subject d has no ses-02A record.
    return pd.DataFrame({'sub_ID': ['a', 'b', 'c'],
                         'age_y2_true': [12.1, 12.0, 11.9]})


def test_correct_y2_age_accuracy():
    out, n_match, n_miss = fix_y2_age.correct_y2_age(_df_y2(), _y2age())
    assert (n_match, n_miss) == (3, 1)
    got = dict(zip(out['sub_ID'], out['interview_age']))
    assert got['a'] == 12.1 and got['b'] == 12.0 and got['c'] == 11.9   # true age used
    assert got['d'] == 10.1                                             # unmatched keeps old age
    assert 'age_y2_true' not in out.columns                            # helper column dropped
    # other columns untouched
    assert list(out['prop_SCAN']) == [0.020, 0.030, 0.025, 0.028]


def test_correct_y2_age_idempotent():
    once, _, _ = fix_y2_age.correct_y2_age(_df_y2(), _y2age())
    twice, _, _ = fix_y2_age.correct_y2_age(once, _y2age())
    pd.testing.assert_frame_equal(once.reset_index(drop=True),
                                  twice.reset_index(drop=True))
