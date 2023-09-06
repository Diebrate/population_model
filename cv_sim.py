import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

import load

import time
start_time = time.time()

np.random.seed(12345)
torch.manual_seed(54321)

for data_name in ['moon', 'wot', 'root']:

    for m in range(100):

        data_origin, T = load.load(data_name)

        nt = data_origin.time.nunique()

        t_list = sorted(data_origin.time.unique())

        pick_ind = np.random.uniform(size=data_origin.shape[0]) < 0.7
        data = data_origin[pick_ind]

        for i in range(1, len(t_list) - 1):

            data_train = data[data.time != t_list[i]]
            data_test = data[data.time == t_list[i]]

            data_train.to_csv('data/cv/' + data_name + '_m' + str(m) + '_t' + str('i') + '_train.csv')
            data_test.to_csv('data/cv/' + data_name + '_m' + str(m) + '_t' + str('i') + '_test.csv')

