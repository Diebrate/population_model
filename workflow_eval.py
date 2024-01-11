import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ot
import os

import load
import nn_framework
import ot_num

m = 0

r_kl_list = [1, 5, 10]
r_lock_list = [1, 5, 10]
r_ent_list = [1, 5, 10]
s_list = [0.01, 0.05, 0.1]

setting_file = pd.read_csv('data/setting/valid_best_param.csv', index_col=0)

best_param = pd.DataFrame(index=['wot', 'root', 'moon'], columns=['TrajectoryNet', 'FBSDE'])

perf = {}

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
                'fb_iter': 20, # 100
                # simulation setting
                'nt': 200,
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

for data_name in ['wot', 'root', 'moon']:

    data, T = load.load(data_name, frac=1)
    N = data.shape[0]

    if data_name == 'wot':
        param_list['n_layers'] = 3
    elif data_name in ['root', 'moon']:
        param_list['n_layers'] = 2

    for method in ['TrajectoryNet', 'FBSDE', 'Waddington-OT']:

        obj = np.inf

        nn_framework.torch.manual_seed(m)
        rng = np.random.default_rng(m)
        idx = rng.permutation(np.arange(N))
        idx_train = idx[:int(0.5 * N)]
        idx_valid = idx[int(0.5 * N):int(0.75 * N)]
        idx_test = idx[int(0.75 * N):]
        data_train = data.iloc[idx_train]
        data_valid = data.iloc[idx_valid]
        data_test = data.iloc[idx_test]

        scaler = StandardScaler()

        data_train.loc[:, ['x', 'y']] = scaler.fit_transform(data_train[['x', 'y']])
        data_valid.loc[:, ['x', 'y']] = scaler.transform(data_valid[['x', 'y']])
        data_test.loc[:, ['x', 'y']] = scaler.transform(data_test[['x', 'y']])

        x0 = data_test[data_test.time == 0][['x', 'y']].sample(param_list['n_test'], replace=True).to_numpy()
        t_check = data.time.unique()
        t_check.sort()
        t_check = t_check[t_check > 0]

        if method in ['TrajectoryNet', 'FBSDE']:

            i_kl, i_lock, i_ent, i_s = [int(i) for i in list(setting_file.loc[data_name, method][-4:])]
            r_kl, r_lock, r_ent, s = r_kl_list[i_kl], r_lock_list[i_lock], r_ent_list[i_ent], s_list[i_s]

            print(f'STARTING---data: {data_name}, method: {method}, r_kl: {r_kl}, r_lock: {r_lock}, r_ent: {r_ent}, s: {s}')
            model_name = f'model/{data_name}_{method}_m{m}_{i_kl}{i_lock}{i_ent}{i_s}.pt'

            wass = []

            res = nn_framework.torch.load(model_name)

            for _ in range(100):
                res_sim = nn_framework.sim_path_soft(res, x0, T=T, t_check=t_check, plot=False, use_gpu=False, **param_list)
                loss = np.zeros(len(t_check))
                for i, t in enumerate(t_check):
                    x_test = res_sim[res_sim.time == t][['x', 'y']].to_numpy()
                    x_ref = data_valid[data_valid.time == t][['x', 'y']].to_numpy()
                    cdist = ot_num.compute_dist(x_test, x_ref, dim=2, single=False)
                    cdist_rev = cdist.copy()
                    px = np.ones(x_test.shape[0]) / x_test.shape[0]
                    py = np.ones(x_ref.shape[0]) / x_ref.shape[0]
                    loss[i] = ot.emd2(px, py, cdist)

        elif method == 'Waddington-OT':

            for i in range(nt_sim - 1):
        ti = t_sim[i]
        tf = t_sim[i + 1]
        if ti == t_trim[ind_check]:
            d0 = data[data.time == t_trim[ind_check]][['x', 'y']].sample(param_list['n_sample'] * 3, replace=False).to_numpy()
            d1 = data[data.time == t_trim[ind_check + 1]][['x', 'y']].sample(param_list['n_sample'] * 3, replace=False).to_numpy()
            costm = ot_num.compute_dist(d0, d1, dim=2, single=False)
            p0 = np.ones(d0.shape[0]) / d0.shape[0]
            p1 = np.ones(d1.shape[0]) / d1.shape[0]
            tmap = ot_num.ot_unbalanced_log_stabilized(p0, p1, costm, param_list['reg'], param_list['reg1'], param_list['reg2'], reg_list)
            tmap = np.diag(1 / tmap.sum(axis=1)) @ tmap
            n0 = d0.shape[0]
            n1 = d1.shape[0]
            ref_ind = np.zeros(n1)
            for j in range(n0):
                ref_ind[j] = np.random.choice(np.arange(n1), p=tmap[j, :])
            ref_ind = ref_ind.astype(int)
            cdist = ot_num.compute_dist(x, d0, dim=2, single=False)
            x_start = x.copy()
            x_end = d1[ref_ind[cdist.argmin(axis=1)]]
            t_start = t_trim[ind_check]
            t_end = t_trim[ind_check + 1]
            ind_check += 1
            gamma = (tf - t_start) / (t_end - t_start)
            x = (1 - gamma) * x_start + gamma * x_end
            res = np.vstack((res, x))


print(best_param)
best_param.to_csv('data/setting/valid_best_param.csv')