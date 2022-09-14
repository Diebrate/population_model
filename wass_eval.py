import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ot_num
import ot

import re
import os

import time
start_time = time.time()

# available time fraction 0.1, 0.2, 0.5, 0.8
method_list = ['disc_ot', 'stat_ot', 'soft', 'fbsde_score']
data_list = ['moon', 'wot', 'root']

perf = {}

from param import *
# edit setting
# M = 3

np.random.seed(54321)

for data_name in data_list:

    if data_name == 'root':
        setting_id = 10
        n_layers = 2
    elif data_name == 'moon':
        setting_id = 10
        n_layers = 2
    elif data_name == 'wot':
        setting_id = 4
        n_layers = 3

    perf[data_name] = np.empty((len(method_list), M), dtype=object)

    for m in range(M):

        for im in range(len(method_list)):

            method = method_list[im]

            if method == 'soft' or method == 'fbsde_score':
                df_sim_name = 'data/sample/m' + str(m) + '_' + data_name + '_' + method + '_sim_full_id' + str(setting_id) + '_l' + str(n_layers) + '.csv'
                df_test_name = 'data/sample/m' + str(m) + '_' + data_name + '_' + method + '_test_full_id' + str(setting_id) + '_l' + str(n_layers) + '.csv'
            elif method == 'disc_ot' or method == 'stat_ot':
                df_sim_name = 'data/sample/m' + str(m) + '_' + data_name + '_' + method + '_sim_full_id' + str(setting_id) + '.csv'
                df_test_name = 'data/sample/m' + str(m) + '_' + data_name + '_' + method + '_test_full_id' + str(setting_id) + '.csv'

            if os.path.exists(df_sim_name) and os.path.exists(df_test_name):

                perf[data_name][im, m] = []
                perf[data_name][im, m] = []

                df_sim = pd.read_csv(df_sim_name, index_col=0)
                df_test = pd.read_csv(df_test_name, index_col=0)

                df_sim = df_sim[['x', 'y', 'time']]
                df_test = df_test[['x', 'y', 'time']]

                t_check = df_test.time.unique()
                t_check = t_check[t_check > 0]
                t_check.sort()

                for t in t_check:
                    print('checking ' + data_name + ' data' + ', method = ' + method + ', m = ' + str(m) + ', t = ' + str(t))
                    x_test = df_sim[df_sim.time == t].drop('time', axis=1).to_numpy()
                    x_ref = df_test[df_test.time == t].drop('time', axis=1).to_numpy()
                    cdist = ot_num.compute_dist(x_test, x_ref, dim=2, single=False)
                    cdist_rev = cdist.copy()
                    # loss = ot_num.loss_balanced_cont(x_test, x_ref, 1, dim=2)
                    px = np.ones(x_test.shape[0]) / x_test.shape[0]
                    py = np.ones(x_ref.shape[0]) / x_ref.shape[0]
                    loss = ot.emd2(px, py, cdist)
                    perf[data_name][im, m].append(loss)
                    # print('finished time frac = ' + str(time_frac) + ', eval method ' + data_name + ', method = ' + method + ', m = ' + str(m) + ', t = ' + str(t))
                perf[data_name][im, m] = np.mean(perf[data_name][im, m])

            else:
                print('file not found at ' + data_name + ' data' + ', method = ' + method + ', m = ' + str(m) + ', t = ' + str(t))
                perf[data_name][im, m] = np.nan

perf_mean = {em: np.nanmean(perf[em].astype(float), axis=1).astype(str) for em in data_list}
perf_var = {em: np.char.add(np.char.add('(', np.nanstd(perf[em].astype(float), axis=1).astype(str)), ')') for em in data_list}

perf = {em: np.char.add(perf_mean[em], perf_var[em]) for em in data_list}
perf = pd.DataFrame(data=perf, index=method_list)

perf.to_csv('data/eval/full_model_summary.csv')

print("--- %s seconds ---" % (time.time() - start_time))