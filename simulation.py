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

save_data = False
use_sys = False
if use_sys:
    import sys
    time_frac = float(sys.argv[1])
    data_name = str(sys.argv[2])
    method = str(sys.argv[3])
    setting_id = int(sys.argv[4])
    n_layers = int(sys.argv[5])
    m = int(sys.argv[6])
else:
    time_frac = 1
    data_name = 'root'
    method = 'disc_ot'
    setting_id = 10
    n_layers = 2
    m = 0

np.random.seed(12345)
torch.manual_seed(54321)

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

data_origin, T = load.load(data_name)

t_full = data_origin.time.unique()
t_full.sort()

t_check = t_full[t_full > 0]

data_all = pd.read_csv('data/dgp/m' + str(m) + '_' + data_name + '_all.csv')

if time_frac != 1:
    t_trim = np.random.choice(t_full[1:-1], size=int(max(1, np.floor(time_frac * (t_full.shape[0] - 2)))), replace=False)
    t_trim = np.concatenate(([0], t_trim, [t_full[-1]]))
    t_trim = np.unique(t_trim)
    t_trim.sort()
    data = data_all[np.isin(data_all.time, t_trim)]
    data_test = data_all[~np.isin(data_all.time, t_trim)]
    t_sim = np.concatenate((np.linspace(0, T, param_list['nt']), t_trim, t_check))
else:
    pick_ind = np.random.uniform(size=data_all.shape[0]) < 0.7
    data = data_all[pick_ind]
    data_test = data_all[~pick_ind]
    t_sim = np.concatenate((np.linspace(0, T, param_list['nt']), t_check))
    t_trim = t_full.copy()

x0 = data[data.time == 0][['x', 'y']].sample(param_list['n_test'], replace=True).to_numpy()

t_sim = np.unique(t_sim)
t_sim.sort()
nt_sim = len(t_sim)

if method == 'ot':
    res = nn_framework.train_alg_mfc_ot(data, T=T, track=True, **param_list)
    res_sim = nn_framework.sim_path_soft(res['model'], x0, T=T, t_check=t_check, plot=True, **param_list)
elif method == 'force':
    res = nn_framework.train_alg_mfc_force(data, T=T, track=True, **param_list)
    res_sim = nn_framework.sim_path_force(res['model'], x0, T=T, data_full=data, t_check=t_check, plot=True, **param_list)
elif method == 'soft':
    res = nn_framework.train_alg_mfc_soft(data, T=T, track=True, **param_list)
    res_sim = nn_framework.sim_path_soft(res['model'], x0, T=T, t_check=t_check, plot=True, **param_list)
elif method == 'fbsde':
    res = nn_framework.train_alg_mfc_fbsde(data, T=T, track=True, **param_list)
    res_sim = nn_framework.sim_path_soft(res['model_f'], x0, T=T, t_check=t_check, plot=True, **param_list)
elif method == 'fbsde_score':
    res = nn_framework.train_alg_mfc_fbsde(data, T=T, track=True, use_score=True, **param_list)
    res_sim = nn_framework.sim_path_soft(res['model_f'], x0, T=T, t_check=t_check, plot=True, **param_list)
elif method == 'soft_seg':
    res = nn_framework.train_alg_mfc_soft_seg(data, T=T, track=True, **param_list)
    res_sim = nn_framework.sim_path_soft_seg(res['model'], x0, T=T, t_check=t_check, bound=res['bound'], plot=True, **param_list)
elif method == 'fb_ot':
    res = nn_framework.train_alg_mfc_fb_ot(data, T=T, track=True, **param_list)
    res_sim = nn_framework.sim_path_fb_ot(res, x0, T=T, t_check=t_check, plot=True, **param_list)
elif method == 'mixed':
    res = nn_framework.train_alg_mfc_mixed(data, T=T, track=True, **param_list)
    res_sim = nn_framework.sim_path_mixed(res, x0, T=T, t_check=t_check, plot=True, **param_list)
elif method == 'fb_mixed':
    res = nn_framework.train_alg_mfc_fb_mixed(data, T=T, track=True, **param_list)
    res_sim = nn_framework.sim_path_mixed(res, x0, T=T, t_check=t_check, fb=True, plot=True, **param_list)
elif method == 'fb_mixed_score':
    res = nn_framework.train_alg_mfc_fb_mixed(data, T=T, track=True, use_score=True, **param_list)
    res_sim = nn_framework.sim_path_mixed(res, x0, T=T, t_check=t_check, fb=True, plot=True, **param_list)
elif method == 'disc_ot':
    x = x0.copy()
    res_sim = x0.copy()
    ind_check = 0
    reg_list = (100 - param_list['reg']) * np.exp(-np.arange(100)) + param_list['reg']
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
        res_sim = np.vstack((res_sim, x))
    res_sim = np.hstack((res_sim, np.repeat(t_sim, param_list['n_test']).reshape(-1, 1)))
    res_sim = pd.DataFrame(res_sim, columns=['x', 'y', 'time'])
elif method == 'stat_ot':
    x = x0.copy()
    res_sim = x0.copy()
    ind_check = 0
    reg_list = (100 - param_list['reg']) * np.exp(-np.arange(100)) + param_list['reg']
    for i in range(nt_sim - 1):
        ti = t_sim[i]
        tf = t_sim[i + 1]
        if ti == t_trim[ind_check]:
            d0 = data[data.time == t_trim[ind_check]][['x', 'y']].to_numpy()
            d1 = data[data.time == t_trim[ind_check + 1]][['x', 'y']].to_numpy()
            costm = ot_num.compute_dist(d0, d1, dim=2, single=False)
            n0 = d0.shape[0]
            n1 = d1.shape[0]
            p0 = np.ones(n0) / n0
            p1 = np.ones(n1) / n1
            tmap = ot_num.ot_unbalanced_log_stabilized(p0, p1, costm, param_list['reg'], param_list['reg1'], param_list['reg2'], reg_list)
            tmap = np.diag(1 / tmap.sum(axis=1)) @ tmap
            d1 = tmap @ d1
            cdist = ot_num.compute_dist(x, d0, dim=2, single=False)
            x_start = x.copy()
            # x_end = d1[cdist.argmin(axis=1)]
            weight = np.exp(-0.5 * cdist)
            weight = np.diag(1 / weight.sum(axis=1)) @ weight
            x_end = weight @ d1
            t_start = t_trim[ind_check]
            t_end = t_trim[ind_check + 1]
            ind_check += 1
        gamma = (tf - t_start) / (t_end - t_start)
        x = (1 - gamma) * x_start + gamma * x_end
        res_sim = np.vstack((res_sim, x))
    res_sim = np.hstack((res_sim, np.repeat(t_sim, param_list['n_test']).reshape(-1, 1)))
    res_sim = pd.DataFrame(res_sim, columns=['x', 'y', 'time'])

res_sim.plot.scatter(x='x', y='y', c='time', cmap='Spectral', s=1, figsize=(10, 8))
# plt.savefig(img_name)

# Get the current figure and axes
fig = plt.gcf()
ax = plt.gca()

# Create a colorbar using the current plot
cbar = ax.collections[0].colorbar
cbar.set_label('time', fontsize=20)

# Modify colorbar properties
cbar.ax.tick_params(labelsize=18)

# Set axis title and ticklabel font properties
ax.set_xlabel('x', fontsize=20)
ax.set_ylabel('y', fontsize=20)
ax.tick_params(labelsize=18)

df_sim_name = 'data/sim/m' + str(m) + '_' + data_name + '_' + method + '_sim_r' + str(time_frac).replace('.', '_') + '.csv'
df_test_name = 'data/sim/m' + str(m) + '_' + data_name +  '_' + method + '_test_r' + str(time_frac).replace('.', '_') + '.csv'
df_train_name = 'data/sim/m' + str(m) + '_' + data_name +  '_' + method + '_train_r' + str(time_frac).replace('.', '_') + '.csv'

if save_data:
    res_sim.to_csv(df_sim_name)
    data_test.to_csv(df_test_name)
    data.to_csv(df_train_name)