import numpy as np
import torch
import pyreadr
import pandas as pd
import matplotlib.pyplot as plt

import nn_framework

data_name = 'circle'
method = 'fb_ot'

np.random.seed(12345)

use_default = True
# This is the default setting for fbsde method
r_v = 1
r_ent = 0.01
r_ent_v = 1
r_kl = 0.001
r_lock = 1
reg = 0.01
reg1 = 50
reg2 = 50
k = 5
lock_dist = 0.001

# model setting
nt_grid = 250
n_seg = 5
n_sample = 100
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
if not use_default:
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
    n_sample = 100
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

img_name = 'image/sim_r' + str(r_v).replace('.', '_') + '_r' +  str(r_ent).replace('.', '_') + '_r' + str(r_kl).replace('.', '_') + '.png' 

print('r_v=' + str(r_v) + '\n' + 'r_ent=' + str(r_ent) + '\n' + 'r_kl=' + str(r_kl))

df_name = 'data/sim_r' + str(r_v).replace('.', '_') + '_r' +  str(r_ent).replace('.', '_') + '_r' + str(r_kl).replace('.', '_') + '.csv' 

print('r_v=' + str(r_v) + '\n' + 'r_ent=' + str(r_ent))

if data_name == 'root':
    # root data
    data_all = pyreadr.read_r('data/data.rds')
    data_all = pd.DataFrame(data_all[None])
    data_all.columns = ['x', 'y', 'time']
    T = data_all.time.max()
elif data_name == 'wot':
    # wot data
    data_all = pd.read_csv('data/df_wot.csv')
    data_all = data_all[['x', 'y', 'day']]
    data_all.columns = ['x', 'y', 'time']
    data_all[['x']] /= 10000
    data_all[['y']] /= 10000
    T = data_all.time.max()
elif data_name == 'syn':
    # synthetic data
    cov = np.array([[1, 0], [0, 1]])
    start = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=5000)
    end1 = np.random.multivariate_normal(mean=[10, 10], cov=cov, size=2500)
    end2 = np.random.multivariate_normal(mean=[10, -10], cov=cov, size=2500)
    data_all = pd.DataFrame(np.vstack((start, end1, end2)))
    data_all.columns = ['x', 'y']
    data_all['time'] = np.repeat([0, 1], 5000)
    T = data_all.time.max()
elif data_name == 'circle':
    # point to ring data
    cov = np.array([[1, 0], [0, 1]])
    start = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=5000) * 0.01
    theta = np.random.uniform(low=0, high=2 * np.pi, size=5000)
    end = np.vstack((10 * np.cos(theta), 10 * np.sin(theta))).T
    data_all = pd.DataFrame(np.vstack((start, end)))
    data_all.columns = ['x', 'y']
    data_all['time'] = np.repeat([0, 1], 5000)
    T = data_all.time.max()
elif data_name == 'spiral':
    # point to ring data
    cov = np.array([[1, 0], [0, 1]])
    sigma = np.sqrt(0.1)
    start = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=5000) * 0.01
    def velo(x1, x2, s=0.5, theta=np.pi * 0.01):
        exp_comp = np.exp(-(np.power(x1, 2) + np.power(x2, 2)) / np.power(s, 2))
        phi = np.array([exp_comp * 2 * x1 / np.power(s, 2) - x1,
                        exp_comp * 2 * x2 / np.power(s, 2) - x2])
        f = 10 * exp_comp * np.array([[np.cos(theta), -np.sin(theta)],
                                      [np.sin(theta), np.cos(theta)]]) @ np.array([[x1],
                                                                                   [x2]]) 
        return phi + f.flatten()
    n_steps = 10
    tau = 1 / n_steps
    time_points = np.linspace(0, 0.01, n_steps + 1)[1:]
    x_temp = start.copy()
    data_all = start.copy()
    for t in time_points:
        v = np.array([velo(*xs) for xs in x_temp])
        x_temp = x_temp + tau * v + sigma * np.sqrt(tau) * np.random.multivariate_normal(mean=[0, 0], cov=cov, size=5000) * 0.01
        data_all = np.vstack((data_all, x_temp))
    data_all = pd.DataFrame(data_all)
    data_all.columns = ['x', 'y']
    data_all['time'] = np.repeat(np.linspace(0, 0.01, n_steps + 1), 5000)
    T = data_all.time.max()

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
    res_sim = nn_framework.sim_path_ot(res, x0, T=T, nt=nt, s1=e_s1, s2=e_s2, plot=True)
elif method == 'force':
    res = nn_framework.train_alg_mfc_force(data, T=T, lr=lr,
                                            n_sample=n_sample, n_iter=n_iter, nt_grid=nt_grid, 
                                            error_s1=e_s1, error_s2=e_s2,
                                            h=h,
                                            r_v=r_v, r_ent=r_ent, r_kl=r_kl,
                                            track=True)
    res_sim = nn_framework.sim_path_force(res['model'], x0, T=T, data_full=data, t_check=t_check, nt=100, s1=e_s1, s2=e_s2, plot=True)
elif method == 'soft':
    res = nn_framework.train_alg_mfc_soft(data, T=T, lr=lr,
                                          n_sample=n_sample, n_iter=n_iter, nt_grid=nt_grid, 
                                          error_s1=e_s1, error_s2=e_s2,
                                          h=h, k=k, lock_dist=lock_dist,
                                          r_v=r_v, r_ent=r_ent, r_kl=r_kl, r_ent_v=r_ent_v, r_lock=r_lock,
                                          track=True)
    res_sim = nn_framework.sim_path_soft(res['model'], x0, T=T, nt=nt, s1=e_s1, s2=e_s2, plot=True)
elif method == 'soft_seg':
    res = nn_framework.train_alg_mfc_soft_seg(data, T=T, lr=lr,
                                              n_sample=n_sample, n_iter=n_iter, nt_grid=nt_grid, n_seg=n_seg, 
                                              error_s1=e_s1, error_s2=e_s2,
                                              h=h, k=k, lock_dist=lock_dist,
                                              r_v=r_v, r_ent=r_ent, r_kl=r_kl, r_ent_v=r_ent_v, r_lock=r_lock,
                                              track=True)
    res_sim = nn_framework.sim_path_soft_seg(res['model'], x0, T=T, nt=nt, bound=res['bound'], s1=e_s1, s2=e_s2, plot=True)
elif method == 'fb_ot':
    res = nn_framework.train_alg_mfc_fb_ot(data, lr=lr,
                                            n_sample=n_sample, n_iter=n_iter, nt_subgrid=nt_subgrid, 
                                            error_s1=e_s1, error_s2=e_s2,
                                            h=h, k=k, lock_dist=lock_dist,
                                            r_v=r_v, r_ent=r_ent, r_kl=r_kl, r_ent_v=r_ent_v, r_lock=r_lock,
                                            reg=reg, reg1=reg1, reg2=reg2,
                                            track=True)
    res_sim = nn_framework.sim_path_fb_ot(res, x0, nt=nt_grid, s1=e_s1, s2=e_s2, h=h, plot=True)

# plt.savefig(img_name)

print('r_v=' + str(r_v) + '\n' + 'r_ent=' + str(r_ent) + '\n' + 'r_kl=' + str(r_kl))

# res_sim.to_csv(df_name)
