import numpy as np

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

M = 20

