import numpy as np
import torch
import pyreadr
import pandas as pd
import matplotlib.pyplot as plt

import nn_framework

np.random.seed(12345)

r_v = 1
r_ent = 1
r_ent_v = 1
r_kl = 15
r_lock = 1
reg = 0.001
reg1 = 50
reg2 = 50

# model setting
nt_grid = 500
n_seg = 5
n_sample = 100

# simulation setting
nt = 500

e_s1 = 0.01
e_s2 = 0.01
# h = None
h = np.diag(np.ones(2)) * 1

img_name = 'image/sim_r' + str(r_v).replace('.', '_') + '_r' +  str(r_ent).replace('.', '_') + '_r' + str(r_kl).replace('.', '_') + '.png' 

print('r_v=' + str(r_v) + '\n' + 'r_ent=' + str(r_ent) + '\n' + 'r_kl=' + str(r_kl))

df_name = 'data/sim_r' + str(r_v).replace('.', '_') + '_r' +  str(r_ent).replace('.', '_') + '_r' + str(r_kl).replace('.', '_') + '.csv' 

print('r_v=' + str(r_v) + '\n' + 'r_ent=' + str(r_ent))

# root data
# data = pyreadr.read_r('data/data.rds')
# data = pd.DataFrame(data[None])
# T = data.time.max()

# wot data
# data = pd.read_csv('data/df_wot.csv')
# data = data[['x', 'y', 'day']]
# data.columns = ['UMAP_1', 'UMAP_2', 'time']
# data[['UMAP_1']] /= 10000
# data[['UMAP_2']] /= 10000
# T = data.time.max()

# synthetic data
cov = np.array([[1, 0], [0, 1]])
start = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=50000)
end1 = np.random.multivariate_normal(mean=[10, 10], cov=cov, size=25000)
end2 = np.random.multivariate_normal(mean=[10, -10], cov=cov, size=25000)
data = pd.DataFrame(np.vstack((start, end1, end2)))
data.columns = ['UMAP_1', 'UMAP_2']
data['time'] = np.repeat([0, 1], 50000)

data = data.sample(50000)
x0 = data[data.time == 0][['UMAP_1', 'UMAP_2']].sample(250, replace=True).to_numpy()

t_check = data.time.unique()
t_check.sort()
t_check = t_check[t_check > 0]

# res = nn_framework.train_alg_mfc_ot(data, T=T, lr=0.001,
#                                     n_sample=n_sample, n_iter=100, nt_grid=nt_grid, 
#                                     error_s1=e_s1, error_s2=e_s2,
#                                     h=h,
#                                     r_v=r_v, r_ent=r_ent, r_ent_v=r_ent_v, r_lock=r_lock,
#                                     reg=reg, reg1=reg1, reg2=reg2,
#                                     track=True)

# res = nn_framework.train_alg_mfc_force(data, T=T, lr=0.001,
#                                         n_sample=100, n_iter=100, nt_grid=nt_grid, 
#                                         error_s1=e_s1, error_s2=e_s2,
#                                         h=h,
#                                         r_v=r_v, r_ent=r_ent, r_kl=r_kl,
#                                         track=True)

# res = nn_framework.train_alg_mfc_soft(data, T=T, lr=0.001,
#                                       n_sample=100, n_iter=150, nt_grid=nt_grid, 
#                                       error_s1=e_s1, error_s2=e_s2,
#                                       h=h,
#                                       r_v=r_v, r_ent=r_ent, r_kl=r_kl, r_ent_v=r_ent_v, r_lock=r_lock,
#                                       track=True)

# res = nn_framework.train_alg_mfc_soft_seg(data, T=T, lr=0.001,
#                                           n_sample=200, n_iter=150, nt_grid=nt_grid, n_seg=n_seg, 
#                                           error_s1=e_s1, error_s2=e_s2,
#                                           h=h,
#                                           r_v=r_v, r_ent=r_ent, r_kl=r_kl, r_ent_v=r_ent_v, r_lock=r_lock,
#                                           track=True)

res = nn_framework.train_alg_mfc_fb_ot(data, lr=0.001,
                                       n_sample=n_sample, n_iter=128, nt_subgrid=5, 
                                       error_s1=1, error_s2=1,
                                       h=h,
                                       r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1,
                                       reg=1, reg1=1, reg2=2,
                                       track=True)

# res_sim = nn_framework.sim_path_ot(res, x0, T=T, nt=nt, s1=e_s1, s2=e_s2, plot=True)

# res_sim = nn_framework.sim_path_force(res['model'], x0, T=T, data_full=data, t_check=t_check, nt=100, s1=e_s1, s2=e_s2, plot=True)

# res_sim = nn_framework.sim_path_soft(res['model'], x0, T=T, nt=nt, s1=e_s1, s2=e_s2, plot=True)

# res_sim = nn_framework.sim_path_soft_seg(res['model'], x0, T=T, nt=nt, bound=res['bound'], s1=e_s1, s2=e_s2, plot=True)

res_sim = nn_framework.sim_path_fb_ot(res, x0, nt=500, s1=1, s2=1, plot=True)

# plt.savefig(img_name)

print('r_v=' + str(r_v) + '\n' + 'r_ent=' + str(r_ent) + '\n' + 'r_kl=' + str(r_kl))

# res_sim.to_csv(df_name)
