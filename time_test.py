import numpy as np
import torch
from torch import nn

import time

np.random.seed(12345)
torch.manual_seed(12345)

x = torch.rand(10000,2)

t0 = time.time()
h = x.T.cov() * (x.shape[0] ** -(1 / 6))
h = h.detach().numpy()
print('cov takes %s  seconds' % (time.time() - t0))

t0 = time.time()
h = np.linalg.inv(h)
print('inv take %s  seconds' % (time.time() - t0))

t0 = time.time()
c1 = x[:, 0].reshape(-1, 1) - x[:, 0]
print('c1 takes %s  seconds' % (time.time() - t0))

t0 = time.time()
c2 = x[:, 1].reshape(-1, 1) - x[:, 1]
print('c2 takes %s seconds' % (time.time() - t0))

t0 = time.time()
c = h[0, 0] * c1.pow(2) + (h[1, 0] + h[0, 1]) * c1 * c2 + h[1, 1] * c2.pow(2)
print('c*h takes %s seconds' % (time.time() - t0))

t0 = time.time()
c_alt = h[0, 0] * c1 * c1 + (h[1, 0] + h[0, 1]) * c1 * c2 + h[1, 1] * c2 * c2
print('c*h alt takes %s seconds' % (time.time() - t0))

t0 = time.time()
c = c1.pow(2) + c2.pow(2)
print('pow sum takes %s seconds' % (time.time() - t0))

t0 = time.time()
c_div = c / 10
print('div takes %s seconds' % (time.time() - t0))

t0 = time.time()
c_t = c.T
print('trans takes %s seconds' % (time.time() - t0))

t0 = time.time()
c_sum = (c1 + c2) / 10
print('c sum takes %s seconds' % (time.time() - t0))

t0 = time.time()
c0 = x[:, None, :] - x
print('out diff takes %s seconds' % (time.time() - t0))

t0 = time.time()
c0 = c0.pow(2).sum(axis=-1)
print('out pow takes %s seconds' % (time.time() - t0))

t0 = time.time()
c = torch.exp(-0.5 * c) #* np.sqrt(h[0, 0] * h[1, 1] - h[0, 1] * h[1, 0]) / (2 * torch.pi)
print('exp takes %s seconds' % (time.time() - t0))




