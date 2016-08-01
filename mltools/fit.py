from __future__ import division, print_function, absolute_import

import pandas as pd
import numpy as np

from scipy import stats
from numpy import linspace

class Weibull():
    """ Use Weibull distribution to fit the data
    and find the critical vaule of X.
    X means the area under the Weibull curve from 0 to X
    """

    def __init__(self, data):
        self.data = np.sort(data)

    def get_params(data, a = 1, c = 1, loc = 1):
        weibull_params = stats.exponweib.fit(data, a, c, loc)
        return weibull_params

    def get_cdf_data(data, params):
        df_cdf = pd.DataFrame({'data':data, 'cdf':stats.exponweib.cdf(data, *params)})
        return df_cdf

    def get_critical_value(df_cdf, cv):
        x = df_cdf[df_cdf['cdf'] > cv ]['data'].values
        cv = x[0]
        return cv
