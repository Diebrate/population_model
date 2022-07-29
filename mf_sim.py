import numpy as np
import pandas as pd
import torch
from torch import nn

import load
from param import param_info
import nn_framework

time_frac = 1.0
data_name = 'root'
method = 'fbsde_score'
setting_id = -2

param_df = pd.read_csv('data/param_multi_setting.csv', index_col=0)
param_list = param_df.iloc[setting_id].to_dict()
if setting_id == -1: # default for wot
    param_list = {#regularizer
                  'r_v': 0.1,
                  'r_ent': 1,
                  'r_ent_v' : 1,
                  'r_kl': 5,
                  'r_lock': 10,
                  'reg': 0.01,
                  'reg1': 50,
                  'reg2': 50,
                  'k': 5,
                  'lock_dist': 0.001,
                  # model setting
                  'nt_grid': 200,
                  'n_seg': 5,
                  'n_sample': 100,
                  'nt_subgrid': 10,
                  'n_mixed': 10,
                  'fb_iter': 100, # 100
                  # simulation setting
                  'nt': 200,
                  'n_test': 200,
                  's1': 0.01,
                  's2': 0.01,
                  'h': 1,
                  # optimization
                  'lr': 0.001,
                  'n_iter': 100,
                  # mc
                  'M': 20,
                  # setting id
                  'setting_id': 0}
elif setting_id == -2: # default for root
    param_list = {#regularizer
                  'r_v': 0.1,
                  'r_ent': 1,
                  'r_ent_v' : 1,
                  'r_kl': 5,
                  'r_lock': 10,
                  'reg': 0.01,
                  'reg1': 50,
                  'reg2': 50,
                  'k': 5,
                  'lock_dist': 0.001,
                  # model setting
                  'nt_grid': 200,
                  'n_seg': 5,
                  'n_sample': 100,
                  'nt_subgrid': 10,
                  'n_mixed': 10,
                  'fb_iter': 100, # 100
                  # simulation setting
                  'nt': 200,
                  'n_test': 200,
                  's1': 0.01,
                  's2': 0.01,
                  'h': 0,
                  # optimization
                  'lr': 0.001,
                  'n_iter': 100,
                  # mc
                  'M': 20,
                  # setting id
                  'setting_id': 0}

for name, info in param_info.items():
    param_list[name] = info(param_list[name])
if param_list['h'] == 0:
    param_list['h'] = None
else:
    param_list['h'] = np.diag(np.ones(2) * param_list['h'])

data_all, T = load.load(data_name)

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

model_name = 'model/' + data_name + '_' + method + '_t' + str(time_frac).replace('.', '_') + '_sim_id' + str(setting_id)
res = {}

if method in ['ot', 'force', 'soft', 'soft_seg']:
    res['model'] = nn_framework.torch.load(model_name + '_model.pt')
    res['optimizer'] = nn_framework.torch.load(model_name + '_opt.pt')
elif method in ['fbsde', 'fbsde_score', 'fb_ot']:
    res['model_f'] = nn_framework.torch.load(model_name + '_model_f.pt')
    res['model_b'] = nn_framework.torch.load(model_name + '_model_b.pt')
    res['optimizer_f'] = nn_framework.torch.load(model_name + '_opt_f.pt')
    res['optimizer_b'] = nn_framework.torch.load(model_name + '_opt_b.pt')
elif method in ['mixed']:
    res['model_drift'] = nn_framework.torch.load(model_name + '_model_drift.pt')
    res['model_mn'] = nn_framework.torch.load(model_name + '_model_mn.pt')
    res['optimizer_drift'] = nn_framework.torch.load(model_name + '_opt_drift.pt')
    res['optimizer_mn'] = nn_framework.torch.load(model_name + '_opt_mn.pt')
elif method in ['fb_mixed', 'fb_mixed_score']:
    res['model_f'] = nn_framework.torch.load(model_name + '_model_f.pt')
    res['model_b'] = nn_framework.torch.load(model_name + '_model_b.pt')
    res['model_f_mn'] = nn_framework.torch.load(model_name + '_model_f_mn.pt')
    res['model_b_mn'] = nn_framework.torch.load(model_name + '_model_b_mn.pt')
    res['optimizer_f'] = nn_framework.torch.load(model_name + '_opt_f.pt')
    res['optimizer_b'] = nn_framework.torch.load(model_name + '_opt_b.pt')
    res['optimizer_f_mn'] = nn_framework.torch.load(model_name + '_opt_f_mn.pt')
    res['optimizer_b_mn'] = nn_framework.torch.load(model_name + '_opt_b_mn.pt')

if method == 'ot':
    res_sim = nn_framework.sim_path_soft(res['model'], x0, T=T, t_check=t_check, plot=True, **param_list)
elif method == 'force':
    res_sim = nn_framework.sim_path_force(res['model'], x0, T=T, data_full=data, t_check=t_check, plot=True, **param_list)
elif method == 'soft':
    res_sim = nn_framework.sim_path_soft(res['model'], x0, T=T, t_check=t_check, plot=True, **param_list)
elif method == 'fbsde':
    res_sim = nn_framework.sim_path_soft(res['model_f'], x0, T=T, t_check=t_check, plot=True, **param_list)
elif method == 'fbsde_score':
    res_sim = nn_framework.sim_path_soft(res['model_f'], x0, T=T, t_check=t_check, plot=True, **param_list)
elif method == 'soft_seg':
    res_sim = nn_framework.sim_path_soft_seg(res['model'], x0, T=T, t_check=t_check, bound=res['bound'], plot=True, **param_list)
elif method == 'fb_ot':
    res_sim = nn_framework.sim_path_fb_ot(res, x0, T=T, t_check=t_check, plot=True, **param_list)
elif method == 'mixed':
    res_sim = nn_framework.sim_path_mixed(res, x0, T=T, t_check=t_check, plot=True, **param_list)
elif method == 'fb_mixed':
    res_sim = nn_framework.sim_path_mixed(res, x0, T=T, t_check=t_check, fb=True, plot=True, **param_list)
elif method == 'fb_mixed_score':
    res_sim = nn_framework.sim_path_mixed(res, x0, T=T, t_check=t_check, fb=True, plot=True, **param_list)
