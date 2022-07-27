import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ot
import load

import time
start_time = time.time()

data_name = 'syn'
time_frac = 1

np.random.seed(12345)

use_default = True

#################################
if use_default:
# This is the default setting for fbsde method
    from param import *
else:
    # trial setting
    r_v = 0.0001
    r_ent = 1
    r_ent_v = 1
    r_kl = 10
    r_lock = 100
    reg = 0.01
    reg1 = 50
    reg2 = 50
    k = 3
    lock_dist = 0.01

    # model setting
    nt_grid = 250
    n_seg = 5
    n_sample = 250
    nt_subgrid = 10

    # simulation setting
    nt = nt_grid
    n_test = 1000

    e_s1 = 0.005
    e_s2 = 0.005
    # h = None
    h = np.diag(np.ones(2)) * 1

    lr = 0.001
    n_iter = 100
#################################

img_name = 'image/' + data_name + '_sim_r' + str(r_v).replace('.', '_') + '_r' +  str(r_ent).replace('.', '_') + '_r' + str(r_kl).replace('.', '_') + '.png'
print(img_name)

df_name = 'data/' + data_name + '_sim_r' + str(r_v).replace('.', '_') + '_r' +  str(r_ent).replace('.', '_') + '_r' + str(r_kl).replace('.', '_') + '.csv'
print(df_name)

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

x = data_test[data_test.time == 0][['x', 'y']].sample(n_test, replace=True).to_numpy()

t_sim = np.concatenate((np.linspace(0, T, nt), t_trim))
t_sim = np.unique(t_sim)
t_sim.sort()
nt_sim = len(t_sim)

res = x.copy()
ind_check = 0
reg_list = (100 - reg) * np.exp(-np.arange(100)) + reg
for i in range(nt_sim - 1):
    ti = t_sim[i]
    tf = t_sim[i + 1]
    if ti == t_trim[ind_check]:
        d0 = data[data.time == t_trim[ind_check]][['x', 'y']].sample(n_sample * 3, replace=False).to_numpy()
        d1 = data[data.time == t_trim[ind_check + 1]][['x', 'y']].sample(n_sample * 3, replace=False).to_numpy()
        costm = ot.compute_dist(d0, d1, dim=2, single=False)
        p0 = np.ones(d0.shape[0]) / d0.shape[0]
        p1 = np.ones(d1.shape[0]) / d1.shape[0]
        tmap = ot.ot_unbalanced_log_stabilized(p0, p1, costm, reg, reg1, reg2, reg_list)
        tmap = np.diag(1 / tmap.sum(axis=1)) @ tmap
        n0 = d0.shape[0]
        n1 = d1.shape[0]
        ref_ind = np.zeros(n1)
        for j in range(n0):
            ref_ind[j] = np.random.choice(np.arange(n1), p=tmap[j, :])
        ref_ind = ref_ind.astype(int)
        cdist = ot.compute_dist(x, d0, dim=2, single=False)
        x_start = x.copy()
        x_end = d1[ref_ind[cdist.argmin(axis=1)]]
        t_start = t_trim[ind_check]
        t_end = t_trim[ind_check + 1]
        ind_check += 1
    gamma = (tf - t_start) / (t_end - t_start)
    x = (1 - gamma) * x_start + gamma * x_end
    res = np.vstack((res, x))

res = np.hstack((res, np.repeat(t_sim, n_test).reshape(-1, 1)))
res = pd.DataFrame(res, columns=['x', 'y', 'time'])

res.plot.scatter(x='x', y='y', c='time', cmap='Spectral', s=1, figsize=(10, 8))
# plt.savefig(img_name)

# res.to_csv(df_name)

print("--- %s seconds ---" % (time.time() - start_time))