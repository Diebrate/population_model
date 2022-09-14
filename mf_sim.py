import numpy as np
import pandas as pd
import torch
from torch import nn

import load
from param import param_info
import nn_framework

import ot_num
import ot

save_model = True
use_sys = True
if use_sys:
    import sys
    time_frac = float(sys.argv[1])
    data_name = str(sys.argv[2])
    method = str(sys.argv[3])
    setting_id = int(sys.argv[4])
    n_layers = int(sys.argv[5])
    m = int(sys.argv[6])
else:
    time_frac = 1.0
    data_name = 'moon'
    method = 'soft'
    setting_id = 10
    n_layers = 2
    m = 15

np.random.seed(12345)

# param_df = pd.read_csv('data/setting/param_multi_setting.csv', index_col=0)
param_df = pd.read_excel('data/setting/param_multi_setting.xlsx', sheet_name=data_name, index_col=0)
param_list = param_df.iloc[setting_id].to_dict()
if data_name == 'wot':
    param_list['nt'] = 100
    param_list['n_test'] = 200
    param_list['s1'] = 0.01
    param_list['s2'] = 0.01
elif data_name == 'root':
    param_list['nt'] = param_list['nt_grid']
    param_list['n_test'] = 200
    param_list['s1'] = 0.01
    param_list['s2'] = 0.01
elif data_name == 'moon':
    param_list['nt'] = param_list['nt_grid']
    param_list['n_test'] = 500
    param_list['s1'] = 0.01
    param_list['s2'] = 0.01

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

model_name = 'model/' + data_name + '_' + method + '_t' + str(time_frac).replace('.', '_') + '_sim_id' + str(setting_id) + '_l' + str(n_layers)
# model_name = 'model/root/' + data_name + '_' + method + '_t' + str(time_frac).replace('.', '_') + '_sim_id' + str(setting_id) + '_l' + str(n_layers)
res = {}

torch.manual_seed(m)

if method in ['ot', 'force', 'soft', 'soft_seg']:
    res['model'] = nn_framework.torch.load(model_name + '_model.pt')
    # res['optimizer'] = nn_framework.torch.load(model_name + '_opt.pt')
elif method in ['fbsde', 'fbsde_score', 'fb_ot']:
    res['model_f'] = nn_framework.torch.load(model_name + '_model_f.pt')
    # res['model_b'] = nn_framework.torch.load(model_name + '_model_b.pt')
    # res['optimizer_f'] = nn_framework.torch.load(model_name + '_opt_f.pt')
    # res['optimizer_b'] = nn_framework.torch.load(model_name + '_opt_b.pt')
elif method in ['mixed']:
    res['model_drift'] = nn_framework.torch.load(model_name + '_model_drift.pt')
    res['model_mn'] = nn_framework.torch.load(model_name + '_model_mn.pt')
    # res['optimizer_drift'] = nn_framework.torch.load(model_name + '_opt_drift.pt')
    # res['optimizer_mn'] = nn_framework.torch.load(model_name + '_opt_mn.pt')
elif method in ['fb_mixed', 'fb_mixed_score']:
    res['model_f'] = nn_framework.torch.load(model_name + '_model_f.pt')
    # res['model_b'] = nn_framework.torch.load(model_name + '_model_b.pt')
    res['model_f_mn'] = nn_framework.torch.load(model_name + '_model_f_mn.pt')
    # res['model_b_mn'] = nn_framework.torch.load(model_name + '_model_b_mn.pt')
    # res['optimizer_f'] = nn_framework.torch.load(model_name + '_opt_f.pt')
    # res['optimizer_b'] = nn_framework.torch.load(model_name + '_opt_b.pt')
    # res['optimizer_f_mn'] = nn_framework.torch.load(model_name + '_opt_f_mn.pt')
    # res['optimizer_b_mn'] = nn_framework.torch.load(model_name + '_opt_b_mn.pt')

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

if save_model:
    df_name = 'data/sample/m' + str(m) + '_' + data_name + '_' + method + '_sim_full_id' + str(setting_id) + '_l' + str(n_layers) + '.csv'
    res_sim.to_csv(df_name)
    df_test_name = 'data/sample/m' + str(m) + '_' + data_name + '_' + method + '_test_full_id' + str(setting_id) + '_l' + str(n_layers) + '.csv'
    data_test.to_csv(df_test_name)

# performance evaluation

t_all = data_all.time.unique()
t_all.sort()
t_all = t_all[t_all > 0]

nt_eval = len(t_all)

check_loss = False

if check_loss:
    wass = np.zeros(nt_eval)
    for ind in range(nt_eval):
        x_test = res_sim[res_sim.time == t_all[ind]].drop('time', axis=1).to_numpy()
        x_ref = data_all[data_all.time == t_all[ind]].drop('time', axis=1).to_numpy()
        cdist = ot_num.compute_dist(x_test, x_ref, dim=2, single=False)
        px = np.ones(x_test.shape[0]) / x_test.shape[0]
        py = np.ones(x_ref.shape[0]) / x_ref.shape[0]
        loss = ot.emd2(px, py, cdist)
        wass[ind] = loss

