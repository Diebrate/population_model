import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import load

from param import M

data_name = 'moon'

use_sys = True
if use_sys:
    import sys
    data_name = str(sys.argv[1])

np.random.seed(12345)

data_origin, T = load.load(data_name, frac=0.5)

for m in range(M):

    data_all = data_origin.copy()
    data_all.x = data_origin.x + np.random.normal(size=data_origin.x.shape[0]) * 0.05
    data_all.y = data_origin.x + np.random.normal(size=data_origin.x.shape[0]) * 0.05

    df_name = 'data/dgp/m' + str(m) + '_' + data_name + '_all.csv'
    data_all.to_csv(df_name)
