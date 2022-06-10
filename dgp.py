import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import load
import nn_framework
import ot

data_name = 'wot'

use_sys = True
if use_sys:
    import sys
    data_name = str(sys.argv[1])

np.random.seed(12345)

use_default = True

#################################
if use_default:
# This is the default setting for fbsde method
    from param import *
else:
    # trial setting
    r_v = 1
    r_ent = 0.01
    r_ent_v = 1
    r_kl = 0.001
    r_lock = 1
    reg = 0.01
    reg1 = 50
    reg2 = 50
    k = 5
    lock_dist = 0.001

    # model setting
    nt_grid = 250
    n_seg = 5
    n_sample = 100
    nt_subgrid = 10

    # simulation setting
    nt = nt_grid
    n_test = 1000

    e_s1 = 0.005
    e_s2 = 0.005
    # h = None
    h = np.diag(np.ones(2)) * 1

    lr = 0.001
    n_iter = 100

    M = 20
#################################

data_origin, T = load.load(data_name, frac=0.5)

for m in range(M):
    
    data_all = data_origin.copy()
    data_all.x = data_origin.x + np.random.normal(size=data_origin.x.shape[0]) * 0.05
    data_all.y = data_origin.x + np.random.normal(size=data_origin.x.shape[0]) * 0.05
    
    df_name = 'data/sim/m' + str(m) + '_' + data_name + '_all.csv' 
    data_all.to_csv(df_name)
