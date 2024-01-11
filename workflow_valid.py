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

best_param = pd.DataFrame(index=['wot', 'root', 'moon'], columns=['TrajectoryNet', 'FBSDE'])

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

    for method in ['TrajectoryNet', 'FBSDE']:

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

        x0 = data_valid[data_valid.time == 0][['x', 'y']].sample(param_list['n_test'], replace=True).to_numpy()
        t_check = data.time.unique()
        t_check.sort()
        t_check = t_check[t_check > 0]

        for i_kl, r_kl in enumerate(r_kl_list):
            for i_lock, r_lock in enumerate(r_lock_list):
                for i_ent, r_ent in enumerate(r_ent_list):
                    for i_s, s in enumerate(s_list):
                        print(f'STARTING---data: {data_name}, method: {method}, r_kl: {r_kl}, r_lock: {r_lock}, r_ent: {r_ent}, s: {s}')
                        model_name = f'model/{data_name}_{method}_m{m}_{i_kl}{i_lock}{i_ent}{i_s}.pt'

                        if os.path.exists(model_name):
                            res = nn_framework.torch.load(model_name)
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
                            if loss.mean() < obj:
                                obj = loss.mean()
                                best_param.loc[data_name, method] = f'm{m}_{i_kl}{i_lock}{i_ent}{i_s}'


print(best_param)
best_param.to_csv('data/setting/valid_best_param.csv')
                                
                