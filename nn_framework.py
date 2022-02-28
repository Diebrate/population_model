import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def kernel(x):
    c = (x[:, 0].reshape(-1, 1) - x[:, 0]).pow(2)
    c = c + (x[:, 1].reshape(-1, 1) - x[:, 1]).pow(2)
    c = torch.exp(-0.5 * c) / torch.tensor(2 * torch.pi)
    return torch.ones(x.shape[0]) @ c / x.shape[0]


def train_alg_est_func(nn_model, x, y, lr=0.05, n_iter=1000):
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=lr)
    L = len(y)
    for n in range(n_iter):
        ind = np.random.choice(np.arange(L))
        pred = model(torch.tensor([[x[ind]]]))
        loss = 0.5 * (torch.tensor([[y[ind]]]) - pred).pow(2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
def train_alg_mfc(nn_model, p0, pt, T, nt=50, n_sample=100, lr=0.005, n_iter=1024):
    dt = T / nt
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=lr)
    mx_ref = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor(p1['mean']), torch.tensor(p1['cov']))
    for n in range(n_iter):
        me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), np.sqrt(dt) * torch.eye(2))
        mx = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor(p0['mean']), torch.tensor(p0['cov']))
        x = mx.sample([n_sample])
        l = torch.zeros(n_sample, requires_grad=True)
        for t in range(nt):
            inp = torch.cat([x, dt * t * torch.ones(n_sample, 1)], dim=1)
            v = nn_model(inp)
            p = kernel(x)
            l = l + 0.5 * torch.linalg.norm(v, axis=1).pow(2) + 0.001 * torch.log(p)
            e = me.sample([n_sample])
            x = x + v + e
            ############################
        p = kernel(x)
        lp_ref = mx_ref.log_prob(x)
        l = 0.01 * (torch.log(p) - lp_ref) @ torch.ones(n_sample, 1) / n_sample
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        
        
def sim_path(nn_model, p0, T, nt=50, n_sample = 100, plot=False):
    dt = T / nt
    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), np.sqrt(dt) * torch.eye(2))
    mx = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor(p0['mean']), torch.tensor(p0['cov']))
    x = mx.sample([n_sample])
    data = []
    data.append(x)
    for n in range(nt):
        inp = torch.cat([x, dt * n * torch.ones(n_sample, 1)], dim=1)
        v = nn_model(inp)
        e = me.sample([n_sample])
        x = x + v + e
        data.append(x)
    t_list = np.hstack((0, [dt * (n + 1) for n in range(nt)]))
    if plot:
        plt.figure(figsize=(10, 8))
        for i in range(nt + 1):
            data_temp = data[i].detach().numpy()
            plt.scatter(data_temp[:, 0], data_temp[:, 1], s=1, label=str(t_list[i]))
    return {'data': data,
            't_list': t_list}
        
        
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
    