import nose
from nose.tools import assert_equal, assert_true

import pandas as pd
import numpy as np
from scipy import stats
from mltools import fit

df_wei = pd.read_csv('data/fit.csv', sep='\t')
df_norm = pd.DataFrame({'value':np.linspace(stats.norm.ppf(0.0001), stats.norm.ppf(0.9999), 100)})

def test_get_cv_wei():
    df = df_wei
    cv1 = fit.get_cv(df, 'Weibull')
    cv2 = fit.get_cv(df, 'weibull')
    cv3 = fit.get_cv(df, 'wei')
    assert_equal(cv1, 1000)
    assert_equal(cv2, 1000)
    assert_equal(cv3, 1000)

def test_get_cv_with_params_wei():
    cv1 = fit.get_cv(df_wei, 'Weibull', cp =0.8)
    assert_equal(cv1, 900)

def test_get_cv_norm():
    df = df_norm
    cv1 = fit.get_cv(df, 'Gaussian')
    cv2 = fit.get_cv(df, 'gaussian')
    cv3 = fit.get_cv(df, 'norm')
    assert_equal(cv1, 3.5687531931140648)
    assert_equal(cv2, 3.5687531931140648)
    assert_equal(cv3, 3.5687531931140648)

def test_get_cv_with_param_norm():
    cv1 = fit.get_cv(df_norm, 'Weibull', cp =0.9999)
    assert_equal(cv1, 3.7190164854557088)
