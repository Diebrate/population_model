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
    

class nn_network:
    
    def __init__(self, L, n_unit, n_in, n_out):
        self.setup(L, n_unit, n_in, n_out)
            
    def setup(self, L, n_unit, n_in, n_out):
        self.L = L
        self.n_unit_per = np.hstack((n_in, np.repeat(n_unit, L), n_out))
        self.W = np.empty(L + 1, dtype=object)
        self.w0 = np.empty(L + 1, dtype=object)
        self.h = np.empty(L + 2, dtype=object)
        self.W_grad = np.empty(L + 1, dtype=object)
        self.w0_grad = np.empty(L + 1, dtype=object)
        for l in range(L + 1):
            self.W[l] = np.random.uniform(-0.5, 0.5, size=(self.n_unit_per[l + 1], self.n_unit_per[l]))
            self.w0[l] = np.random.uniform(-1, 1, size=self.n_unit_per[l + 1])
            self.W_grad[l] = np.zeros((self.n_unit_per[l + 1], self.n_unit_per[l]))
            self.w0_grad[l] = np.zeros(self.n_unit_per[l + 1])
        
    def relu(self, x):
        if len(x) == 1:
            return np.max([0, x])
        else:
            return np.array([np.max([0, x0]) for x0 in x])
    
    def relu_grad(self, x):
        if np.size(x) == 1:
            return 1 if x > 0 else 0
        else:
            return np.array([1 if x0 > 0 else 0 for x0 in x])
        
    def arctan(self, x):
        return np.arctan(x)
    
    def arctan_grad(self, x):
        return 1 / (x ** 2 + 1)
    
    def phi(self, x, f='relu'):
        if f == 'relu':
            return self.relu(x)
        elif f == 'arctan':
            return self.arctan(x)
        
    def phi_grad(self, x, f='relu'):
        if f == 'relu':
            return self.relu_grad(x)
        elif f == 'arctan':
            return self.arctan_grad(x)
    
    def forward(self, h0, f='relu'):
        self.h[0] = h0
        for l in range(self.L):
            if np.size(self.h[l]) == 1:
                self.h[l + 1] = self.phi((self.W[l] * self.h[l]).reshape((-1,)) + self.w0[l], f=f)
            else:
                self.h[l + 1] = self.phi(self.W[l] @ self.h[l] + self.w0[l], f=f)
        self.h[self.L + 1] = self.W[self.L] @ self.h[self.L] + self.w0[self.L]
        
    def backward(self, grad_out, f='relu'):
        self.W_grad[-1] = np.outer(grad_out, self.h[-2])
        self.w0_grad[-1] = grad_out.reshape((-1, ))
        ind = np.arange(self.L)[::-1]
        grad_back = np.reshape(grad_out, (1, -1))
        for l in ind:
            grad_back = grad_back @ self.W[l + 1]
            grad_back = self.phi_grad(self.h[l], f=f) * grad_back
            self.W_grad[l] = np.outer(grad_back, self.h[l])
            self.w0_grad[l] = grad_back.reshape((-1, ))
            
    def grad_descent(self, rate):
        self.W -= rate * self.W_grad
        self.w0 -= rate * self.w0_grad
            
    def test_scenario(self):
        x = np.random.uniform(-np.pi, np.pi, 100)
        e = np.random.normal(loc=0, scale=0.1, size=100)
        y = np.sin(x) + e
        self.setup(3, 10, 1, 1)
        f = 'arctan'
        obj = []
        for m in range(1000):
            yhat = np.zeros(100)
            W_grad = self.W_grad
            w0_grad = self.w0_grad
            for i in range(100):
                self.forward(x[i], f=f)
                yhat[i] = self.h[-1]
            obj.append(0.5 * np.mean((y - yhat) ** 2))
            ind = np.random.choice(np.arange(100))
            self.forward(x[ind], f=f)
            self.backward(self.h[-1] - y[ind], f=f)
            self.grad_descent(0.05)
        plt.figure()
        plt.plot(obj)
        self.test_x = x
        self.test_e = e
        self.test_y = y
        self.test_yhat = yhat
        self.test_obj = obj
        plt.figure()
        plt.scatter(x, y, c='r', s=1)
        plt.scatter(x, yhat, c='b', s=1)
        
        
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
        
# nn_test = nn_network(3, 10, 1, 1)
# nn_test.test_scenario()

# model = NeuralNetwork(1, 1, 20)
# x = np.random.uniform(-np.pi, np.pi, 100)
# e = np.random.normal(loc=0, scale=1, size=100)
# y = x ** 2 + e
# train_alg_est_func(model, x, y, lr=0.01, n_iter=1000)
# pred = model(torch.from_numpy(x).reshape([100, 1])).detach().numpy().reshape((100, ))
# plt.scatter(x, pred, s=1, c='red')
# plt.scatter(x, y, s=1, c='blue')

# mx = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.eye(2))
# x = mx.sample([100])
# p = kernel(x)
# p_ref = torch.exp(mx.log_prob(x))

nn_test = NeuralNetwork(3, 2, 100)
T = 1
p0 = {'mean': [-10., 0.],
      'cov': [[1., 0.],
              [0., 1.]]}
p1 = {'mean': [10., 0.],
      'cov': [[1., 0.],
              [0., 1.]]}
train_alg_mfc(nn_test, p0, p1, 1)
res = sim_path(nn_test, p0, T, nt=50, n_sample = 100, plot=True)