import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import nn_framework
import load

from param import param_info

import time
start_time = time.time()

save_model = True
use_sys = True
if use_sys:
    import sys
    time_frac = float(sys.argv[1])
    data_name = str(sys.argv[2])
    method = str(sys.argv[3])
    setting_id = int(sys.argv[4])
    n_layers = int(sys.argv[5])
else:
    time_frac = 1
    data_name = 'moon'
    method = 'soft'
    setting_id = 10
    n_layers = 2

np.random.seed(12345)
torch.manual_seed(54321)

# param_df = pd.read_csv('data/param_multi_setting.csv', index_col=0)
param_df = pd.read_excel('data/param_multi_setting.xlsx', sheet_name=data_name, index_col=0)
param_list = param_df.iloc[setting_id].to_dict()
if setting_id == -1: # default for wot
    param_list = {#regularizer
                  'r_v': 0.1,
                  'r_ent': 1,
                  'r_ent_v' : 1,
                  'r_kl': 5, # 5
                  'r_lock': 10, # 10
                  'reg': 0.01,
                  'reg1': 50,
                  'reg2': 50,
                  'k': 5,
                  'lock_dist': 0.001,
                  # model setting
                  'nt_grid': 200, # 300
                  'n_seg': 5,
                  'n_sample': 100, # 100
                  'nt_subgrid': 10,
                  'n_mixed': 10,
                  'fb_iter': 100, # 100
                  # simulation setting
                  'nt': 100,
                  'n_test': 100,
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
                  'r_kl': 5, # 5
                  'r_lock': 10, # 10
                  'reg': 0.01,
                  'reg1': 50,
                  'reg2': 50,
                  'k': 5,
                  'lock_dist': 0.001, # 0.01
                  # model setting
                  'nt_grid': 300, # 300
                  'n_seg': 5,
                  'n_sample': 100, # 100
                  'nt_subgrid': 10,
                  'n_mixed': 10,
                  'fb_iter': 100, # 100
                  # simulation setting
                  'nt': 200,
                  'n_test': 100,
                  's1': 0.01,
                  's2': 0.01,
                  'h': 10, # 10
                  # optimization
                  'lr': 0.001,
                  'n_iter': 100,
                  # mc
                  'M': 20,
                  # setting id
                  'setting_id': 0}
elif setting_id == -3: # default for moon
    param_list = {#regularizer
                  'r_v': 0.1,
                  'r_ent': 1,
                  'r_ent_v' : 1,
                  'r_kl': 5, # 5
                  'r_lock': 10, # 10
                  'reg': 0.01,
                  'reg1': 50,
                  'reg2': 50,
                  'k': 5,
                  'lock_dist': 0, # 0.01
                  # model setting
                  'nt_grid': 30, # 300
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
                  'setting_id': 0}
for name, info in param_info.items():
    param_list[name] = info(param_list[name])
if param_list['h'] == 0:
    param_list['h'] = None
# else:
#     param_list['h'] = np.diag(np.ones(2) * param_list['h'])
param_list['n_layers'] = n_layers
param_list['n_iter'] = 1000

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

model_name = 'model/' + data_name + '_' + method + '_t' + str(time_frac).replace('.', '_') + '_sim_id' + str(setting_id) + '_l' + str(n_layers)

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

if save_model:

    save_name = 'image/sim/' + data_name + '_' + method + '_t' + str(time_frac).replace('.', '_') + '_sim_id' + str(setting_id) + '_l' + str(n_layers) + '.png'
    plt.savefig(save_name)

    df_name = 'data/sim/' + data_name + '_' + method + '_t' + str(time_frac).replace('.', '_') + '_sim_id' + str(setting_id) + '_l' + str(n_layers) + '.csv'
    res_sim.to_csv(df_name)

    if method in ['ot', 'force', 'soft', 'soft_seg']:
        nn_framework.torch.save(res['model'], model_name + '_model.pt')
        nn_framework.torch.save(res['optimizer'], model_name + '_opt.pt')
    elif method in ['fbsde', 'fbsde_score', 'fb_ot']:
        nn_framework.torch.save(res['model_f'], model_name + '_model_f.pt')
        nn_framework.torch.save(res['model_b'], model_name + '_model_b.pt')
        nn_framework.torch.save(res['optimizer_f'], model_name + '_opt_f.pt')
        nn_framework.torch.save(res['optimizer_b'], model_name + '_opt_b.pt')
    elif method in ['mixed']:
        nn_framework.torch.save(res['model_drift'], model_name + '_model_drift.pt')
        nn_framework.torch.save(res['model_mn'], model_name + '_model_mn.pt')
        nn_framework.torch.save(res['optimizer_drift'], model_name + '_opt_drift.pt')
        nn_framework.torch.save(res['optimizer_mn'], model_name + '_opt_mn.pt')
    elif method in ['fb_mixed', 'fb_mixed_score']:
        nn_framework.torch.save(res['model_f'], model_name + '_model_f.pt')
        nn_framework.torch.save(res['model_b'], model_name + '_model_b.pt')
        nn_framework.torch.save(res['model_f_mn'], model_name + '_model_f_mn.pt')
        nn_framework.torch.save(res['model_b_mn'], model_name + '_model_b_mn.pt')
        nn_framework.torch.save(res['optimizer_f'], model_name + '_opt_f.pt')
        nn_framework.torch.save(res['optimizer_b'], model_name + '_opt_b.pt')
        nn_framework.torch.save(res['optimizer_f_mn'], model_name + '_opt_f_mn.pt')
        nn_framework.torch.save(res['optimizer_b_mn'], model_name + '_opt_b_mn.pt')

# data_all.loc[(data_all.time == 0)|(data_all.time == T)].plot.scatter('x', 'y', xlim=(-10, 40), ylim=(-10, 40), s=1, c='time', cmap='Spectral')

print("--- %s seconds ---" % (time.time() - start_time))
