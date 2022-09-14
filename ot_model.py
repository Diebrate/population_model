import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ot_num
import ot
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
    m = int(sys.argv[5])
else:
    time_frac = 1.0
    data_name = 'wot'
    method = 'disc_ot'
    setting_id = 4
    m = 1

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


img_name = 'image/sample/m' + str(m) + '_' + data_name + '_sim_full_id' + str(setting_id) + '.png'

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
# x0 = data[data.time == 0][['x', 'y']].sample(param_list['n_test'], replace=True).to_numpy()
x = x0.copy()

t_sim = np.concatenate((np.linspace(0, T, param_list['nt']), t_trim))
t_sim = np.unique(t_sim)
t_sim.sort()
nt_sim = len(t_sim)

res = x.copy()
ind_check = 0
reg_list = (100 - param_list['reg']) * np.exp(-np.arange(100)) + param_list['reg']

np.random.seed(m)

if method == 'disc_ot':
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
elif method == 'stat_ot':
    for i in range(nt_sim - 1):
        ti = t_sim[i]
        tf = t_sim[i + 1]
        if ti == t_trim[ind_check]:
            d0 = data[data.time == t_trim[ind_check]][['x', 'y']].sample(param_list['n_sample'] * 3, replace=False).to_numpy()
            d1 = data[data.time == t_trim[ind_check + 1]][['x', 'y']].sample(param_list['n_sample'] * 3, replace=False).to_numpy()
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
        res = np.vstack((res, x))

res = np.hstack((res, np.repeat(t_sim, param_list['n_test']).reshape(-1, 1)))
res = pd.DataFrame(res, columns=['x', 'y', 'time'])

res.plot.scatter(x='x', y='y', c='time', cmap='Spectral', s=1, figsize=(10, 8))
# plt.savefig(img_name)

if save_model:
    df_name = 'data/sample/m' + str(m) + '_' + data_name + '_' + method + '_sim_full_id' + str(setting_id) + '.csv'
    res.to_csv(df_name)
    df_test_name = 'data/sample/m' + str(m) + '_' + data_name + '_' + method + '_test_full_id' + str(setting_id) + '.csv'
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
        x_test = res[res.time == t_all[ind]].drop('time', axis=1).to_numpy()
        x_ref = data_all[data_all.time == t_all[ind]].drop('time', axis=1).to_numpy()
        cdist = ot_num.compute_dist(x_test, x_ref, dim=2, single=False)
        px = np.ones(x_test.shape[0]) / x_test.shape[0]
        py = np.ones(x_ref.shape[0]) / x_ref.shape[0]
        loss = ot.emd2(px, py, cdist)
        wass[ind] = loss
    print(wass.mean())

print("--- %s seconds ---" % (time.time() - start_time))