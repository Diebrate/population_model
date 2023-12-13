import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import load
import nn_framework


m = 0
data_name = 'root'
method = 'FBSDE'

data, T = load.load(data_name, frac=1)

N = data.shape[0]

nn_framework.torch.manual_seed(m)
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
r_ent_list = [1, 5, 10]
s_list = [0.01, 0.05, 0.1]

i_kl = 1
i_lock = 2
i_ent = 0
i_s = 0

r_kl = r_kl_list[i_kl]
r_lock = r_lock_list[i_lock]
r_ent = r_ent_list[i_ent]
s = s_list[i_s]

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

if data_name == 'wot':
    param_list['n_layers'] = 3
elif data_name in ['root', 'moon']:
    param_list['n_layers'] = 2

if method == 'TrajectoryNet':
    param_list['n_iter'] = 1000

if data_name == 'root':
    param_list['h'] = 1

param_list['r_kl'] = r_kl
param_list['r_locl'] = r_lock
param_list['r_ent'] = r_ent
param_list['s1'] = s
param_list['s2'] = s

model_name = f'model/{data_name}_{method}_m{m}_{i_kl}{i_lock}{i_ent}{i_s}.pt'

x0 = data_test[data_test.time == 0][['x', 'y']].sample(param_list['n_test'], replace=True).to_numpy()
t_check = data.time.unique()
t_check.sort()
t_check = t_check[t_check > 0]

print(f'started r_kl = {r_kl}, r_lock = {r_lock}, r_ent = {r_ent}, s = {s}')

if method == 'TrajectoryNet':
    res = nn_framework.train_alg_mfc_soft(data_train, T=T, track=True, **param_list)
    nn_framework.torch.save(res['model'], model_name)
    res_sim = nn_framework.sim_path_soft(res['model'], x0, T=T, t_check=t_check, plot=True, use_gpu=False, **param_list)
elif method == 'FBSDE':
    res = nn_framework.train_alg_mfc_fbsde(data_train, T=T, track=True, use_score=True, **param_list)
    nn_framework.torch.save(res['model_f'], model_name)
    res_sim = nn_framework.sim_path_soft(res['model_f'], x0, T=T, t_check=t_check, plot=True, use_gpu=False, **param_list)

print(f'completed r_kl = {r_kl}, r_lock = {r_lock}, r_ent = {r_ent}, s = {s}')