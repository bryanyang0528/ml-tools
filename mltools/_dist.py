from __future__ import division, print_function, absolute_import
import pandas as pd
import numpy as np
from scipy import stats

class distribution(object):

    def __init__(self, data, **kwds):
        self.data = np.sort(data)
        self.cp = kwds.get('cp', 0.95)
        #params
        #df_cdf

    def get_params(self):
        return self.params

    def get_cdf_data(self):
        return self.df_cdf

    def get_cv(self):
        """ cp means cumulative probability
        """
        df_cdf = self.df_cdf
        cp = self.cp
        x = df_cdf.loc[(df_cdf['cdf'] > cp), "data"].values
        if len(x) > 0:
            cv = x[0]
        else:
            cv=self.data.max()
        return cv

class Weibull(distribution):
    """ Use Weibull distribution to fit the data
    and find the critical vaule of X.
    X means the area under the Weibull curve from 0 to X
    """

    def __init__(self, data, **kwds):
        super(Weibull, self).__init__(data, **kwds)
        self.a = kwds.get('a', 1)
        self.c = kwds.get('c', 1)
        self.floc = kwds.get('floc', 0)
        self.params = stats.exponweib.fit(self.data, self.a, self.c, floc = self.floc)
        self.df_cdf = pd.DataFrame({'data':self.data, 'cdf':stats.exponweib.cdf(self.data, *self.params)})

