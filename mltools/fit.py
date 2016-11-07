from __future__ import division, print_function, absolute_import

import pandas as pd
from ._dist import Weibull

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
