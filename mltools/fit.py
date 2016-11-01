from __future__ import division, print_function, absolute_import

import pandas as pd
import numpy as np

from scipy import stats
from numpy import linspace

weibull = ['Weibull', 'weibull', 'wei']


def get_first_col(data):
    if isinstance(data, pd.core.frame.DataFrame):
        data = data[data.columns[0]]
        return data
    else:
        raise TypeError('Please put a pandas Dataframe')

def get_cv(data, method, **kwds):
    data = get_first_col(data)
    cv = None

    if method in weibull:
        model = Weibull(data, **kwds)
        cv = model.get_cv()
    
    return cv


class Weibull():
    """ Use Weibull distribution to fit the data
    and find the critical vaule of X.
    X means the area under the Weibull curve from 0 to X
    """

    def __init__(self, data, **kwds):
        self.data = np.sort(data)
        self.a = kwds.get('a', 1)
        self.c = kwds.get('c', 1)
        self.floc = kwds.get('floc', 0)
        self.cp = kwds.get('cp', 0.95)
        self.weibull_params = stats.exponweib.fit(self.data, self.a, self.c, floc = self.floc)
        self.df_cdf = pd.DataFrame({'data':self.data, 'cdf':stats.exponweib.cdf(self.data, *self.weibull_params)})

    def get_params(self):
        return self.weibull_params

    def get_cdf_data(self):
        return self.df_cdf

    def get_cv(self):
        """ cp means cumulative probability
        """
        df_cdf = self.df_cdf
        cp = self.cp
        x = df_cdf.loc[(df_cdf['cdf'] > cp), "data"].values
        cv = x[0]
        return cv
