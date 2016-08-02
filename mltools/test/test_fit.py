import nose
from nose.tools import assert_equal, assert_true

import pandas as pd
from mltools import fit 

df = pd.read_csv('data/fit.csv', sep='\t')


def test_get_cv():
    cv1 = fit.get_cv(df, 'Weibull')
    cv2 = fit.get_cv(df, 'weibull')
    cv3 = fit.get_cv(df, 'wei')
    assert_equal(cv1, 1000)
    assert_equal(cv2, 1000)
    assert_equal(cv3, 1000)

def test_get_cv_with_params():
    cv1 = fit.get_cv(df, 'Weibull', cp =0.8)
    assert_equal(cv1, 900)
