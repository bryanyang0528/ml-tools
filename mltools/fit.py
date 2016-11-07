from __future__ import division, print_function, absolute_import

import pandas as pd
from ._dist import *

dist_name = {'weibull':['Weibull', 'weibull', 'wei'],
             'gaussian':['Gaussian', 'gaussian', 'gau', 'norm'] }

def get_first_col(data):
    if isinstance(data, pd.core.frame.DataFrame):
        pass
    elif isinstance(data, pandas.core.series.Series):
        data = pd.DataFrame(data)
    else:
        raise TypeError('Please put a pandas Dataframe')

    data = data[data.columns[0]]
    return data

def get_cv(data, method, **kwds):
    data = get_first_col(data)
    cv = None

    if method in dist_name['weibull']:
        model = Weibull(data, **kwds)
        cv = model.get_cv()
    elif method in dist_name['gaussian']:
        model = Gaussian(data, **kwds)
        cv = model.get_cv()

    return cv
