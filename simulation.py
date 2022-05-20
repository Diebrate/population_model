import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import load
import nn_framework
import ot

data_name = 'root'
method = 'fb_ot'

use_sys = True
if use_sys:
    import sys
    time_frac = float(sys.argv[1])
    data_name = str(sys.argv[2])
else:
    time_frac = 0.8

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

data_origin, T = load.load(data_name, frac=0.5)

t_full = data_origin.time.unique()
t_trim = np.random.choice(t_full, size=int(time_frac * t_full.shape[0]), replace=False)
t_trim = np.concatenate(([0], t_trim, [t_full.max()]))
t_trim = np.unique(t_trim)
t_trim.sort()

t_check = data_origin.time.unique()
t_check.sort()
t_check = t_check[t_check > 0]

for m in range(M):
    
    data_all = data_origin.copy()
    data_all.x = data_origin.x + np.random.normal(size=data_origin.x.shape[0]) * 0.05
    data_all.y = data_origin.x + np.random.normal(size=data_origin.x.shape[0]) * 0.05
    
    data = data_all[np.isin(data_all.time, t_trim)]
    data_test = data_all[~np.isin(data_all.time, t_trim)]
    x0 = data[data.time == 0][['x', 'y']].sample(n_test, replace=True).to_numpy()
    
    if method == 'ot':
        res = nn_framework.train_alg_mfc_ot(data, T=T, lr=lr,
                                            n_sample=n_sample, n_iter=n_iter, nt_grid=nt_grid, 
                                            error_s1=e_s1, error_s2=e_s2,
                                            h=h,
                                            r_v=r_v, r_ent=r_ent, r_ent_v=r_ent_v, r_lock=r_lock,
                                            reg=reg, reg1=reg1, reg2=reg2,
                                            track=False)
        res_mf = nn_framework.sim_path_ot(res, x0, T=T, t_check=t_check, nt=nt, s1=e_s1, s2=e_s2, plot=False)
    elif method == 'force':
        res = nn_framework.train_alg_mfc_force(data, T=T, lr=lr,
                                                n_sample=n_sample, n_iter=n_iter, nt_grid=nt_grid, 
                                                error_s1=e_s1, error_s2=e_s2,
                                                h=h,
                                                r_v=r_v, r_ent=r_ent, r_kl=r_kl,
                                                track=False)
        res_mf = nn_framework.sim_path_force(res['model'], x0, T=T, data_full=data, t_check=t_check, nt=nt, s1=e_s1, s2=e_s2, plot=False)
    elif method == 'soft':
        res = nn_framework.train_alg_mfc_soft(data, T=T, lr=lr,
                                              n_sample=n_sample, n_iter=n_iter, nt_grid=nt_grid, 
                                              error_s1=e_s1, error_s2=e_s2,
                                              h=h, k=k, lock_dist=lock_dist,
                                              r_v=r_v, r_ent=r_ent, r_kl=r_kl, r_ent_v=r_ent_v, r_lock=r_lock,
                                              track=False)
        res_mf = nn_framework.sim_path_soft(res['model'], x0, T=T, t_check=t_check, nt=nt, s1=e_s1, s2=e_s2, plot=False)
    elif method == 'soft_seg':
        res = nn_framework.train_alg_mfc_soft_seg(data, T=T, lr=lr,
                                                  n_sample=n_sample, n_iter=n_iter, nt_grid=nt_grid, n_seg=n_seg, 
                                                  error_s1=e_s1, error_s2=e_s2,
                                                  h=h, k=k, lock_dist=lock_dist,
                                                  r_v=r_v, r_ent=r_ent, r_kl=r_kl, r_ent_v=r_ent_v, r_lock=r_lock,
                                                  track=False)
        res_mf = nn_framework.sim_path_soft_seg(res['model'], x0, T=T, t_check=t_check, nt=nt, bound=res['bound'], s1=e_s1, s2=e_s2, plot=False)
    elif method == 'fb_ot':
        res = nn_framework.train_alg_mfc_fb_ot(data, lr=lr,
                                                n_sample=n_sample, n_iter=n_iter, nt_subgrid=nt_subgrid, 
                                                error_s1=e_s1, error_s2=e_s2,
                                                h=h, k=k, lock_dist=lock_dist,
                                                r_v=r_v, r_ent=r_ent, r_kl=r_kl, r_ent_v=r_ent_v, r_lock=r_lock,
                                                reg=reg, reg1=reg1, reg2=reg2,
                                                track=False)
        res_mf = nn_framework.sim_path_fb_ot(res, x0, t_check=t_check, nt=nt, s1=e_s1, s2=e_s2, h=h, plot=False)
    
    t_sim = np.concatenate((np.linspace(0, T, nt), t_trim, t_check))
    t_sim = np.unique(t_sim)
    t_sim.sort()
    nt_sim = len(t_sim)
    
    x = x0.copy()
    res_ot = x0.copy()
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
        res_ot = np.vstack((res_ot, x))
        
    res_ot = np.hstack((res_ot, np.repeat(t_sim, n_test).reshape(-1, 1)))
    res_ot = pd.DataFrame(res_ot, columns=['x', 'y', 'time'])
    
    res_mf['source'] = 'mf'
    res_ot['source'] = 'ot'
    
    res_comp = pd.concat([res_mf, res_ot], ignore_index=True)

    df_sim_name = 'data/sim/m' + str(m) + '_' + data_name + '_sim_r' + str(time_frac).replace('.', '_') + '.csv'
    df_test_name = 'data/sim/m' + str(m) + '_' + data_name + '_test_r' + str(time_frac).replace('.', '_') + '.csv'
    df_train_name = 'data/sim/m' + str(m) + '_' + data_name + '_train_r' + str(time_frac).replace('.', '_') + '.csv' 
    res_comp.to_csv(df_sim_name)
    data_test.to_csv(df_test_name)
    data.to_csv(df_train_name)