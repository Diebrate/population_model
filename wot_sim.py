import numpy as np
import pyreadr
import pandas as pd
import matplotlib.pyplot as plt

import ot

reg = 0.1
reg1 = 1
reg2 = 50

img_name = 'image/wot_sim_r' + str(reg).replace('.', '_') + '_r' + str(reg1).replace('.', '_') + '_r' + str(reg2).replace('.', '_') + '.png'
print(img_name)

df_name = 'data/wot_sim_r' + str(reg).replace('.', '_') + '_r' + str(reg1).replace('.', '_') + '_r' + str(reg2).replace('.', '_') + '.csv'
print(df_name)

data = pyreadr.read_r('data/data.rds')
data = pd.DataFrame(data[None])
T = 50

data = data.sample(66666)
x0 = data[data.time == 0][['UMAP_1', 'UMAP_2']].to_numpy()
times = data.time.unique()
times.sort()

tmap = []
reg = 0.1
reg_list = (100 - reg) * np.exp(-np.arange(100)) + reg

n_tmap = times.size - 1

x = []
x.append(np.random.choice(np.arange(x0.shape[0]), size=100))
wot = x0[x[-1]]

for ind in range(n_tmap):
    t0 = times[ind]
    t1 = times[ind + 1]
    d0 = data[data.time == t0][['UMAP_1', 'UMAP_2']].to_numpy()
    d1 = data[data.time == t1][['UMAP_1', 'UMAP_2']].to_numpy()
    costm = ot.compute_dist(d0, d1, dim=1, single=False)
    n0 = d0.shape[0]
    n1 = d1.shape[0]
    p0 = np.ones(n0) / n0
    p1 = np.ones(n1) / n1
    tmap.append(ot.ot_unbalanced_log_stabilized(p0, p1, costm, reg, reg1, reg2, reg_list))
    tnorm = np.diag(1 / tmap[-1].sum(axis=1)) @ tmap[-1]
    x_temp = np.array([np.random.choice(np.arange(n1), p=tnorm[i, :]) for i in x[-1]])
    x.append(x_temp)
    wot = np.vstack((wot, d1[x[-1]]))
    
wot = np.hstack((wot, np.repeat(times, 100).reshape(-1, 1)))
wot = pd.DataFrame(wot, columns=['x', 'y', 'time'])

# wot.plot.scatter(x='x', y='y', c='time', cmap='Spectral', s=1, figsize=(10, 8))
# plt.savefig(img_name)

wot.to_csv(df_name)
