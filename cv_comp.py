import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

import load
import nn_framework
import ot_num

from param import M
from param import param_info

import time
start_time = time.time()

np.random.seed(12345)
torch.manual_seed(54321)

save_model = False
use_sys = False
if use_sys:
    import sys
    time_frac = float(sys.argv[1])
    data_name = str(sys.argv[2])
    method = str(sys.argv[3])
    m = int(sys.argv[4])
    t = int(sys.argv[5])
    if len(sys.argv) > 6:
        use_gpu = int(sys.argv[6]) == 1
    else:
        use_gpu = False
else:
    time_frac = 1.0
    data_name = 'root'
    method = 'soft'
    m = 0
    t = 1
    use_gpu = False

if data_name == 'moon':
    setting_id = 10
    n_layers = 2
elif data_name == 'wot':
    setting_id = 4
    n_layers = 3
elif data_name == 'root':
    setting_id = 10
    n_layers = 2

# param_df = pd.read_csv('data/setting/param_multi_setting.csv', index_col=0)
param_df = pd.read_excel('data/setting/param_multi_setting.xlsx', sheet_name=data_name, index_col=0)
param_list = param_df.iloc[setting_id].to_dict()

for name, info in param_info.items():
    param_list[name] = info(param_list[name])
if param_list['h'] == 0:
    param_list['h'] = None
# else:
#     param_list['h'] = np.diag(np.ones(2) * param_list['h'])
param_list['n_layers'] = n_layers

data_train = pd.read_csv('data/cv/' + data_name + '_m' + str(m) + '_t' + str('i') + '_train.csv')
data_test = pd.read_csv('data/cv/' + data_name + '_m' + str(m) + '_t' + str('i') + '_test.csv')

x0 = data_train[data_train.time == 0][['x', 'y']].sample(param_list['n_test'], replace=True).to_numpy()

T = data_train.time.max()
t_check = sorted(np.concatenate([data_train.time.unique(), data_test.time.unique()]))
t_sim = np.concatenate((np.linspace(0, T, param_list['nt']), t_check))
nt_sim = len(t_sim)
t_trim = t_check.copy()

if use_gpu:

    if method == 'soft':
        res = nn_framework.train_alg_mfc_soft_gpu(data_train, T=T, track=True, **param_list)
        res_sim = nn_framework.sim_path_soft(res['model'], x0, T=T, t_check=t_check, plot=True, use_gpu=True, **param_list)
    elif method == 'fbsde_score':
        res = nn_framework.train_alg_mfc_fbsde_gpu(data_train, T=T, track=True, use_score=True, **param_list)
        res_sim = nn_framework.sim_path_soft(res['model_f'], x0, T=T, t_check=t_check, plot=True, use_gpu=True, **param_list)

else:

    if method == 'soft':
        res = nn_framework.train_alg_mfc_soft(data_train, T=T, track=True, **param_list)
        res_sim = nn_framework.sim_path_soft(res['model'], x0, T=T, t_check=t_check, plot=True, **param_list)
    elif method == 'fbsde_score':
        res = nn_framework.train_alg_mfc_fbsde(data_train, T=T, track=True, use_score=True, **param_list)
        res_sim = nn_framework.sim_path_soft(res['model_f'], x0, T=T, t_check=t_check, plot=True, **param_list)

if method == 'disc_ot':
    x = x0.copy()
    res_sim = x0.copy()
    ind_check = 0
    reg_list = (100 - param_list['reg']) * np.exp(-np.arange(100)) + param_list['reg']
    for i in range(nt_sim - 1):
        ti = t_sim[i]
        tf = t_sim[i + 1]
        if ti == t_trim[ind_check]:
            d0 = data_train[data_train.time == t_trim[ind_check]][['x', 'y']].sample(param_list['n_sample'] * 3, replace=False).to_numpy()
            d1 = data_train[data_train.time == t_trim[ind_check + 1]][['x', 'y']].sample(param_list['n_sample'] * 3, replace=False).to_numpy()
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
        res_sim = np.vstack((res_sim, x))
    res_sim = np.hstack((res_sim, np.repeat(t_sim, param_list['n_test']).reshape(-1, 1)))
    res_sim = pd.DataFrame(res_sim, columns=['x', 'y', 'time'])

res_sim.to_csv('data/cv/' + data_name + '_m' + str(m) + '_t' + str('i') + '_' + method + '.csv')

print("--- %s seconds ---" % (time.time() - start_time))

