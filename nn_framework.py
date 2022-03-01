import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn


def kernel(x, s1=1, s2=1):
    c = (x[:, 0].reshape(-1, 1) - x[:, 0]).pow(2) / s1
    c = c + (x[:, 1].reshape(-1, 1) - x[:, 1]).pow(2) / s2
    c = torch.exp(-0.5 * c) / torch.tensor(2 * torch.pi * np.sqrt(s1 * s2))
    return c.mean(axis=0)


def kernel_pred(x, y, s1=1, s2=1):
    c = (x[:, 0].reshape(-1, 1) - y[:, 0]).pow(2) / s1
    c = c + (x[:, 1].reshape(-1, 1) - y[:, 1]).pow(2) / s2
    c = torch.exp(-0.5 * c) / torch.tensor(2 * torch.pi * np.sqrt(s1 * s2))
    return c.mean(axis=0)


def train_alg_est_func(nn_model, x, y, lr=0.05, n_iter=1000):
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=lr)
    L = len(y)
    for n in range(n_iter):
        ind = np.random.choice(np.arange(L))
        pred = nn_model(torch.tensor([[x[ind]]]))
        loss = 0.5 * (torch.tensor([[y[ind]]]) - pred).pow(2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
def train_alg_mfc(data, T, lr=0.001,
                  n_sample=200, n_iter=128, nt_frac=0.1, 
                  error_s1=1, error_s2=1,
                  kernel_s1=1, kernel_s2=1,
                  r_v=0.01, r_ent=0.1, r_kl=10,
                  track=False):
    
    t_list = data.time.unique()
    t_list.sort()
    nt = t_list.size
    dt = 1 / nt
    nt_batch = np.floor(nt_frac * nt)
    if nt_batch == 0:
      nt_batch = 1
    batch_num = nt // nt_batch
    batch_num = int(batch_num)
    
    model = np.empty(batch_num, dtype=object)
    optimizer = np.empty(batch_num, dtype=object)
    cost = np.empty(batch_num, dtype=object)
    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.tensor(dt * np.diag([error_s1, error_s2])).float())
    
    t_start = []
    
    for b in range(batch_num):
        
        model[b] = NeuralNetwork(3, 2, 100)
        optimizer[b] = torch.optim.SGD(model[b].parameters(), lr=lr)
        t0 = int(b * nt_batch)
        t_start.append(T * t0 / nt)
        
        if b == batch_num - 1:
            tf = int(nt - 1)
        else:
            tf = int(t0 + nt_batch - 1)
        
        x0 = data[data.time == t_list[t0]]
        
        x_tensor = []
        
        for t in range(t0, tf + 1):
            x_tensor.append(torch.from_numpy(data[data.time == t_list[t]][['UMAP_1','UMAP_2']].to_numpy()))
            
        obj = []
        
        for n in range(n_iter):
            x = torch.from_numpy(x0.sample(n_sample, replace=True)[['UMAP_1','UMAP_2']].to_numpy())
            l = torch.zeros(n_sample, requires_grad=True)
            for t in range(t0, tf):
                inp = torch.cat([x, t_list[t] * torch.ones(n_sample, 1) / T], dim=1)
                v = model[b](inp)
                e = me.sample([n_sample])
                x = x + v * dt + e
                # x = x + v * dt
                # individual cost
                l = l + r_v * 0.5 * v.pow(2).sum(axis=1) * dt
                # kernel density estimation
                phat = kernel(x, s1=kernel_s1, s2=kernel_s2)
                p = kernel_pred(x_tensor[t - t0], x, s1=kernel_s1, s2=kernel_s2)
                # entropy/preferrence for crowdedness
                l = l + r_ent * torch.log(p) * dt
                # penalty term
                l = l + r_kl * ((t + 1 - t0) / (tf + 1 - t0)) * (phat.log() - p.log())
            l = l.mean()
            optimizer[b].zero_grad()
            l.backward()
            optimizer[b].step()
            obj.append(float(l))
            if bool(l.isnan()):
                raise ArithmeticError('encountered nan at iteration ' + str(int(n)) + ' batch ' + str(int(b)))
            if track:
                print('c = ', str(float(l)), ' at iteration ', str(int(n)), ' batch ', str(int(b)))
            
        cost[b] = obj
        
    return {'model': model,
            'optimizer': optimizer,
            'cost': cost,
            't_list': t_start}
        
        
def sim_path(model, x0, T, t_list, nt=50, s1=1, s2=2, plot=False):
    
    def find_ind_m(t, t_refs):
        k = len(t_refs)
        search = True
        ind = 0
        while ind < k - 1 and search:
            if t >= t_refs[ind] and t < t_refs[ind + 1]:
                search = False
            else:
                ind += 1
        return ind    
        
    dt = 1 / nt
    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.tensor(dt * np.diag([s1, s2])).float())
    data = x0
    ts = [0]
    x = torch.tensor(x0)
    n_sample = x.shape[0]
    for t_ind in range(nt):
        t = T * t_ind / nt
        ind_m = find_ind_m(t, t_list)
        inp = torch.cat([x, t_ind * torch.ones(n_sample, 1) / nt], dim=1)
        v = model[ind_m](inp)
        e = me.sample([n_sample])
        x = x + v * dt + e
        data = np.vstack((data, x.detach().numpy()))
        ts.append(t)
    if plot:
        data = pd.DataFrame(data, columns=['x', 'y'])
        data['time'] = np.repeat(ts, n_sample)
        data.plot.scatter(x='x', y='y', c='time', s=1, cmap='Spectral', figsize=(10, 8))
    return data
        
        
class NeuralNetwork(nn.Module):
    
    def __init__(self, d_in, d_out, d_hid):
        super().__init__()
        self.flatten = nn.Flatten()
        self.flow = nn.Sequential(nn.Linear(d_in, d_hid),
                                  nn.ReLU(),
                                  nn.Linear(d_hid, d_hid),
                                  nn.Tanh(),
                                  nn.Linear(d_hid, d_hid),
                                  nn.ReLU(),
                                  nn.Linear(d_hid, d_out))
        
    def forward(self, x):
        x = self.flatten(x)
        return self.flow(x.float())
    