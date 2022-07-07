import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nn_framework
import load


use_sys = True
if use_sys:
    import sys
    time_frac = float(sys.argv[1])
    data_name = str(sys.argv[2])
    method = str(sys.argv[3])
else:
    time_frac = 1
    data_name = 'root'
    method = 'mixed'

np.random.seed(12345)

use_default = False

#################################
if use_default:
# This is the default setting for fbsde method
    from param import *
else:
    # trial setting
    r_v = 1
    r_ent = 1
    r_ent_v = 1
    r_kl = 1
    r_lock = 1
    reg = 0.01
    reg1 = 50
    reg2 = 50
    k = 5
    lock_dist = 0.001

    # model setting
    nt_grid = 128
    n_seg = 5
    n_sample = 100
    nt_subgrid = 10
    n_mixed = 4

    # simulation setting
    nt = nt_grid
    n_test = 1000

    e_s1 = 0.001
    e_s2 = 0.001
    h = None
    # h = np.diag(np.ones(2)) * 1

    lr = 0.001
    n_iter = 128

    M = 20
#################################

if use_sys:
    n_mixed = int(sys.argv[4])

img_name = 'image/' + data_name + '_' + method + '_sim_r' + str(r_v).replace('.', '_') + '_r' +  str(r_ent).replace('.', '_') + '_r' + str(r_kl).replace('.', '_') + '.png' 

print('r_v=' + str(r_v) + '\n' + 'r_ent=' + str(r_ent) + '\n' + 'r_kl=' + str(r_kl))

df_name = 'data/' + data_name + '_sim_r' + str(r_v).replace('.', '_') + '_r' +  str(r_ent).replace('.', '_') + '_r' + str(r_kl).replace('.', '_') + '.csv' 

print('r_v=' + str(r_v) + '\n' + 'r_ent=' + str(r_ent))

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
x0 = data_test[data_test.time == 0][['x', 'y']].sample(n_test, replace=True).to_numpy()

t_check = data.time.unique()
t_check.sort()
t_check = t_check[t_check > 0]

if method == 'ot':
    res = nn_framework.train_alg_mfc_ot(data, T=T, lr=lr,
                                        n_sample=n_sample, n_iter=n_iter, nt_grid=nt_grid, 
                                        error_s1=e_s1, error_s2=e_s2,
                                        h=h,
                                        r_v=r_v, r_ent=r_ent, r_ent_v=r_ent_v, r_lock=r_lock,
                                        reg=reg, reg1=reg1, reg2=reg2,
                                        track=True)
    res_sim = nn_framework.sim_path_ot(res, x0, T=T, t_check=t_check, nt=nt, s1=e_s1, s2=e_s2, plot=True)
elif method == 'force':
    res = nn_framework.train_alg_mfc_force(data, T=T, lr=lr,
                                            n_sample=n_sample, n_iter=n_iter, nt_grid=nt_grid, 
                                            error_s1=e_s1, error_s2=e_s2,
                                            h=h,
                                            r_v=r_v, r_ent=r_ent, r_kl=r_kl,
                                            track=True)
    res_sim = nn_framework.sim_path_force(res['model'], x0, T=T, data_full=data, t_check=t_check, nt=nt, s1=e_s1, s2=e_s2, plot=True)
elif method == 'soft':
    res = nn_framework.train_alg_mfc_soft(data, T=T, lr=lr,
                                          n_sample=n_sample, n_iter=n_iter, nt_grid=nt_grid, 
                                          error_s1=e_s1, error_s2=e_s2,
                                          h=h, k=k, lock_dist=lock_dist,
                                          r_v=r_v, r_ent=r_ent, r_kl=r_kl, r_ent_v=r_ent_v, r_lock=r_lock,
                                          track=True)
    res_sim = nn_framework.sim_path_soft(res['model'], x0, T=T, t_check=t_check, nt=nt, s1=e_s1, s2=e_s2, plot=True)
elif method == 'soft_seg':
    res = nn_framework.train_alg_mfc_soft_seg(data, T=T, lr=lr,
                                              n_sample=n_sample, n_iter=n_iter, nt_grid=nt_grid, n_seg=n_seg, 
                                              error_s1=e_s1, error_s2=e_s2,
                                              h=h, k=k, lock_dist=lock_dist,
                                              r_v=r_v, r_ent=r_ent, r_kl=r_kl, r_ent_v=r_ent_v, r_lock=r_lock,
                                              track=True)
    res_sim = nn_framework.sim_path_soft_seg(res['model'], x0, T=T, t_check=t_check, nt=nt, bound=res['bound'], s1=e_s1, s2=e_s2, plot=True)
elif method == 'fb_ot':
    res = nn_framework.train_alg_mfc_fb_ot(data, lr=lr,
                                            n_sample=n_sample, n_iter=n_iter, nt_subgrid=nt_subgrid, 
                                            error_s1=e_s1, error_s2=e_s2,
                                            h=h, k=k, lock_dist=lock_dist,
                                            r_v=r_v, r_ent=r_ent, r_kl=r_kl, r_ent_v=r_ent_v, r_lock=r_lock,
                                            reg=reg, reg1=reg1, reg2=reg2,
                                            track=True)
    res_sim = nn_framework.sim_path_fb_ot(res, x0, t_check=t_check, nt=nt, s1=e_s1, s2=e_s2, h=h, plot=True)
elif method == 'mixed':
    res = nn_framework.train_alg_mfc_mixed(data, T=T, lr=lr, M=n_mixed,
                                           n_sample=n_sample, n_iter=n_iter, nt_grid=nt_grid, 
                                           error_s1=e_s1, error_s2=e_s2,
                                           h=h, k=k, lock_dist=lock_dist,
                                           r_v=r_v, r_ent=r_ent, r_kl=r_kl, r_ent_v=r_ent_v, r_lock=r_lock,
                                           track=True)
    res_sim = nn_framework.sim_path_mixed(res, x0, T=T, t_check=t_check, nt=nt, s1=e_s1, s2=e_s2, plot=True)

# plt.savefig(img_name)

print('r_v=' + str(r_v) + '\n' + 'r_ent=' + str(r_ent) + '\n' + 'r_kl=' + str(r_kl))

# res_sim.to_csv(df_name)

save_name = 'image/' + data_name + '_' + method + '_t' + str(time_frac).replace('.', '_') + '_sim_m' + str(n_mixed) + '.png'
# plt.savefig(save_name)