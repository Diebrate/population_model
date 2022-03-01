import numpy as np
import torch
import pyreadr
import pandas as pd
import matplotlib.pyplot as plt

import nn_framework

data = pyreadr.read_r('data/data.rds')
data = pd.DataFrame(data[None])

data = data.sample(50000)
T = 50

x0 = data[data.time == 0][['UMAP_1', 'UMAP_2']].to_numpy()
e_s1 = 0.5
e_s2 = 0.5
k_s1 = 0.5
k_s2 = 0.5

res = nn_framework.train_alg_mfc(data, T=50, lr=0.01,
                  n_sample=200, n_iter=512, nt_frac=0.2, 
                  error_s1=e_s1, error_s2=e_s2,
                  kernel_s1=k_s1, kernel_s2=k_s2, 
                  r_v=0.05, r_ent=0.25, r_kl=15, track=True)

data.plot.scatter(x='UMAP_1', y='UMAP_2', c='time', cmap='Spectral', s=1, figsize=(10, 8))
plt.savefig('image/true.png')
nn_framework.sim_path(res['model'], x0, T=50, s1=e_s1, s2=e_s2, t_list=res['t_list'], plot=True)
plt.savefig('image/sim.png')
