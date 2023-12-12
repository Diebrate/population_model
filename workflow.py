import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys

import load

m = int(sys.argv[1])
data_name = str(sys.argv[2])

data = load.load(data_name, frac=1)

N = data.shape[0]

rng = np.random.default_rng(m)
mask = np.zeros(N)
mask[rng.choice(np.arange(N), size=int(0.7 * N), replace=False)] = 1
data_train = data.iloc[mask == 1]
data_test = data.iloc[mask == 0]

scaler = StandardScaler()

data_train.loc[:, ['x', 'y']] = scaler.fit_transform(data_train[['x', 'y']])
data_test.loc[:, ['x', 'y']] = scaler.fit_transform(data_test[['x', 'y']])

r_kl_list = [1, 5, 10]
r_lock_list = [1, 5, 10]
r_ent_list = [0.1, 0.5, 1]
lock_dist = 0.1
s_list = [0.01, 0.1, 1]

param_list = {
                # regularizer
                'r_v': 0.1,
                'r_ent': 1,
                'r_ent_v' : 0,
                'r_kl': 5, # 5
                'r_lock': 10, # 10
                'reg': 0.01,
                'reg1': 50,
                'reg2': 50,
                'k': 5,
                'lock_dist': 0.1, # 0.01
                # model setting
                'nt_grid': 200, # 300
                'n_seg': 5,
                'n_sample': 100, # 100
                'nt_subgrid': 10,
                'n_mixed': 10,
                'fb_iter': 100, # 100
                # simulation setting
                'nt': 30,
                'n_test': 100,
                's1': 0.01,
                's2': 0.01,
                'h': 1, # 10
                # optimization
                'lr': 0.001,
                'n_iter': 100,
                # mc
                'M': 20,
                # setting id
                'setting_id': 0
             }

for r_kl in r_kl_list:
    for r_lock in r_lock_list:
        for r_ent in r_ent_list:
            for s in s_list:
                t_full = data_all.time.unique()
                t_trim = np.random.choice(t_full, size=int(time_frac * t_full.shape[0]), replace=False)
                t_trim = np.concatenate(([0], t_trim, [t_full.max()]))
                t_trim = np.unique(t_trim)
                t_trim.sort()
                data_all = data_all[np.isin(data_all.time, t_trim)]

                data_all = data_all.sample(frac=0.7, replace=False)
                n = int(data_all.shape[0])
                n_train = int(n * 0.9)
                ind_all = np.random.permutation(np.arange(n))

                data = data_all.iloc[ind_all[:n_train]]
                data_test = data_all.iloc[ind_all[n_train:]]
                x0 = data_test[data_test.time == 0][['x', 'y']].sample(param_list['n_test'], replace=True).to_numpy()

                t_check = data.time.unique()
                t_check.sort()
                t_check = t_check[t_check > 0]

                model_name = 'model/' + data_name + '_' + method + '_t' + str(time_frac).replace('.', '_') + '_sim_id' + str(setting_id) + '_l' + str(n_layers)

                if method == 'TrajectoryNet':
                    res = nn_framework.train_alg_mfc_soft_gpu(data, T=T, track=True, **param_list)
                    res_sim = nn_framework.sim_path_soft(res['model'], x0, T=T, t_check=t_check, plot=True, use_gpu=True, **param_list)
                elif method == 'FBSDE':
                    res = nn_framework.train_alg_mfc_fbsde_gpu(data, T=T, track=True, use_score=True, **param_list)
                    res_sim = nn_framework.sim_path_soft(res['model_f'], x0, T=T, t_check=t_check, plot=True, use_gpu=True, **param_list)
