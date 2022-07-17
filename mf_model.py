import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nn_framework
import load

from param import param_info

use_sys = True
if use_sys:
    import sys
    time_frac = float(sys.argv[1])
    data_name = str(sys.argv[2])
    method = str(sys.argv[3])
    setting_id = int(sys.argv[4])
else:
    time_frac = 1
    data_name = 'wot'
    method = 'fbsde'
    setting_id = 0

np.random.seed(12345)

param_df = pd.read_csv('data/param_multi_setting.csv', index_col=0)
param_list = param_df.iloc[setting_id].to_dict()
# param_list = {#regularizer
#               'r_v': 0.01,
#               'r_ent': 0.01,
#               'r_ent_v' : 1,
#               'r_kl': 0.1,
#               'r_lock': 5,
#               'reg': 0.01,
#               'reg1': 50,
#               'reg2': 50,
#               'k': 10,
#               'lock_dist': 0.001,
#               # model setting
#               'nt_grid': 200,
#               'n_seg': 5,
#               'n_sample': 200,
#               'nt_subgrid': 10,
#               'n_mixed': 10,
#               'fb_iter': 10,
#               # simulation setting
#               'nt': 200,
#               'n_test': 1000,
#               's1': 0.05,
#               's2': 0.05,
#               'h': 0,
#               # optimization
#               'lr': 0.001,
#               'n_iter': 128,
#               # mc
#               'M': 20,
#               # setting id
#               'setting_id': 0}
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

if method == 'ot':
    res = nn_framework.train_alg_mfc_ot(data, T=T, track=True, **param_list)
    res_sim = nn_framework.sim_path_ot(res, x0, t_check=t_check, plot=True, **param_list)
elif method == 'force':
    res = nn_framework.train_alg_mfc_force(data, T=T, track=True, **param_list)
    res_sim = nn_framework.sim_path_force(res['model'], x0, T=T, data_full=data, t_check=t_check, plot=True, **param_list)
elif method == 'soft':
    res = nn_framework.train_alg_mfc_soft(data, T=T, track=True, **param_list)
    res_sim = nn_framework.sim_path_soft(res['model'], x0, T=T, t_check=t_check, plot=True, **param_list)
elif method == 'fbsde':
    res = nn_framework.train_alg_mfc_fbsde(data, T=T, track=True, **param_list)
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

# plt.savefig(img_name)
# res_sim.to_csv(df_name)

save_name = 'image/sim/' + data_name + '_' + method + '_t' + str(time_frac).replace('.', '_') + '_sim_id' + str(setting_id) + '.png'
plt.savefig(save_name)