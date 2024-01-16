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

param_list = {
                # regularizer
                'r_v': 0.1,
                'r_ent': 1,
                'r_ent_v' : 0,
                'r_kl': 5, # 5
                'r_lock': 10, # 10
                'reg': 1,
                'reg1': 1,
                'reg2': 1,
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

data_list = ['wot', 'root', 'moon']
method_list = ['TrajectoryNet', 'FBSDE', 'Waddington-OT']
perf = pd.DataFrame(np.zeros((len(data_list), len(method_list))), index=data_list, columns=method_list)

for data_name in data_list:

    data, T = load.load(data_name, frac=1)
    N = data.shape[0]

    if data_name == 'wot':
        param_list['n_layers'] = 3
    elif data_name in ['root', 'moon']:
        param_list['n_layers'] = 2

    i_kl, i_lock, i_ent, i_s = [int(i) for i in list(setting_file.loc[data_name, 'FBSDE'][-4:])]
    r_kl, r_lock, r_ent, s = r_kl_list[i_kl], r_lock_list[i_lock], r_ent_list[i_ent], s_list[i_s]

    for method in method_list:

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

        wass = []

        print(f'STARTING---------data: {data_name}, method: {method}---------')

        if method in ['TrajectoryNet', 'FBSDE']:

            # i_kl, i_lock, i_ent, i_s = [int(i) for i in list(setting_file.loc[data_name, method][-4:])]
            # r_kl, r_lock, r_ent, s = r_kl_list[i_kl], r_lock_list[i_lock], r_ent_list[i_ent], s_list[i_s]

            model_name = f'model/{data_name}_{method}_m{m}_{i_kl}{i_lock}{i_ent}{i_s}.pt'

            res = nn_framework.torch.load(model_name)

            for _ in range(100):
                res_sim = nn_framework.sim_path_soft(res, x0, T=T, t_check=t_check, plot=False, use_gpu=False, **param_list)
                loss = np.zeros(len(t_check))
                for i, t in enumerate(t_check):
                    x_test = res_sim[res_sim.time == t][['x', 'y']].to_numpy()
                    x_ref = data_test[data_test.time == t][['x', 'y']].to_numpy()
                    cdist = ot_num.compute_dist(x_test, x_ref, dim=2, single=False)
                    cdist_rev = cdist.copy()
                    px = np.ones(x_test.shape[0]) / x_test.shape[0]
                    py = np.ones(x_ref.shape[0]) / x_ref.shape[0]
                    loss[i] = ot.emd2(px, py, cdist)
                wass.append(loss.mean())

        elif method == 'Waddington-OT':

            t_all = [0] + t_check
            num_time = len(t_all)
            reg = param_list['reg']
            reg1 = param_list['reg1']
            reg2 = param_list['reg2']

            for _ in range(100):
                loss = []
                for i in range(1, num_time - 1):
                    t = t_all[i]
                    # ti = t_all[i - 1]
                    # tf = t_all[i + 1]
                    ti = t_all[max(0, i - 5)]
                    tf = t_all[min(num_time - 1, i + 5)]
                    d0 = data_test[data_test.time == ti][['x', 'y']].sample(param_list['n_sample'], replace=False).to_numpy()
                    d1 = data_test[data_test.time == tf][['x', 'y']].sample(param_list['n_sample'], replace=False).to_numpy()
                    costm = ot_num.compute_dist(d0, d1, dim=2, single=False)
                    p0 = np.ones(d0.shape[0]) / d0.shape[0]
                    p1 = np.ones(d1.shape[0]) / d1.shape[0]
                    tmap = ot_num.ot_unbalanced(p0, p1, costm, reg, reg1, reg2)
                    nd0 = d0.shape[0]
                    nd1 = d1.shape[0]
                    ind_pair = np.random.choice(np.arange(nd0 * nd1), size=param_list['n_sample'], replace=True, p=tmap.flatten())
                    x_start = d0[ind_pair // nd1]
                    x_end = d1[ind_pair % nd1]
                    gamma = (t - ti) / (tf - ti)
                    x_test = (1 - gamma) * x_start + gamma * x_end
                    x_ref = data_test[data_test.time == t][['x', 'y']].to_numpy()
                    cdist = ot_num.compute_dist(x_test, x_ref, dim=2, single=False)
                    cdist_rev = cdist.copy()
                    px = np.ones(x_test.shape[0]) / x_test.shape[0]
                    py = np.ones(x_ref.shape[0]) / x_ref.shape[0]
                    loss.append(ot.emd2(px, py, cdist))
                wass.append(np.mean(loss))

        perf.loc[data_name, method] = f'{np.nanmean(wass):.3f} +- {np.nanstd(wass):.3f}'

perf.to_csv('data/eval/valid_summary.csv')