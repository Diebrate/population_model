# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadr


data_true = pyreadr.read_r('data/data.rds')
data_true = pd.DataFrame(data_true[None])
data_true.columns = ['x', 'y', 'time']
data_true['source'] = 'real'
T = data_true.time.max()

r_v = 0.1
r_ent = 0.1
r_kl = 5

df_name = 'data/sim_r' + str(r_v).replace('.', '_') + '_r' +  str(r_ent).replace('.', '_') + '_r' + str(r_kl).replace('.', '_') + '.csv' 

data_sim = pd.read_csv(df_name, index_col=0)
data_sim.columns = ['x', 'y', 'time']
data_sim['source'] = 'sim'

data = pd.concat([data_true, data_sim], axis=0)
data.reset_index(drop=True, inplace=True)

# for i in range(10):
#     plt.figure(figsize=(10, 8))
#     sns.scatterplot(data=data[data.time==i],x='x',y='y',hue='source', edgecolor=None)

n_sample = data_sim.groupby('time').mean('time')

sample = data_sim.groupby('time').head(1)
sample.plot.scatter('x', 'y', c='time', cmap='Spectral', figsize=(10, 8))
plt.plot(sample['x'], sample['y'])