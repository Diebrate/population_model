import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.linalg

import torch
from torch import nn

import ot_num


def tensor_cost(x, y):
    c1 = x[:, 0].reshape(-1, 1) - y[:, 0]
    c2 = x[:, 1].reshape(-1, 1) - y[:, 1]
    c = c1.pow(2) + c2.pow(2)
    return c


def score_est(x, y, h=None):
    if h is None:
        h = x.T.cov() * (x.shape[0] ** (-1 / 6))
        # h = x.T.cov() / x.shape[0]
        h = h.detach().numpy()
    else:
        h = np.diag([h, h])
    h = np.linalg.inv(h)
    c1 = x[:, 0].reshape(-1, 1) - y[:, 0]
    c2 = x[:, 1].reshape(-1, 1) - y[:, 1]
    c = h[0, 0] * c1.pow(2) + (h[0, 1] + h[1, 0]) * c1 * c2 + h[1, 1] * c2.pow(2)
    c = torch.exp(-0.5 * c)
    score_num = torch.rand(y.shape[0], 2)
    score_num[:, 0] = ((h[0, 0] * c1 + 0.5 * (h[0, 1] + h[1, 0]) * c2) * c).sum(axis=0)
    score_num[:, 1] = ((h[1, 1] * c2 + 0.5 * (h[0, 1] + h[1, 0]) * c1) * c).sum(axis=0)
    score_denom = torch.diag(1 / c.sum(axis=0))
    return score_denom.float() @ score_num.float()


def score_est_gpu(x, y, h=None):
    if h is None:
        h = x.T.cov() * (x.shape[0] ** (-1 / 6))
        # h = x.T.cov() / x.shape[0]
        h = h.cpu().detach().numpy()
    else:
        h = np.diag([h, h])
    h = np.linalg.inv(h)
    h = torch.tensor(h).cuda()
    c1 = x[:, 0].reshape(-1, 1) - y[:, 0]
    c2 = x[:, 1].reshape(-1, 1) - y[:, 1]
    c = h[0, 0] * c1.pow(2) + (h[0, 1] + h[1, 0]) * c1 * c2 + h[1, 1] * c2.pow(2)
    c = torch.exp(-0.5 * c)
    score_num = torch.rand(y.shape[0], 2).cuda()
    score_num[:, 0] = ((h[0, 0] * c1 + 0.5 * (h[0, 1] + h[1, 0]) * c2) * c).sum(axis=0)
    score_num[:, 1] = ((h[1, 1] * c2 + 0.5 * (h[0, 1] + h[1, 0]) * c1) * c).sum(axis=0)
    score_denom = torch.diag(1 / c.sum(axis=0))
    return score_denom.float() @ score_num.float()


def kernel(x, h=None):
    c1 = x[:, 0].reshape(-1, 1) - x[:, 0]
    c2 = x[:, 1].reshape(-1, 1) - x[:, 1]
    if h is None:
        h = x.T.cov() * (x.shape[0] ** (-1 / 6))
        # h = x.T.cov() / x.shape[0]
        h = h.detach().numpy()
        h = np.linalg.inv(h)
        c = h[0, 0] * c1.pow(2) + (h[0, 1] + h[1, 0]) * c1 * c2 + h[1, 1] * c2.pow(2)
        # c = torch.exp(-0.5 * c) * np.sqrt(h[0, 0] * h[1, 1] - h[0, 1] * h[1, 0])/ torch.tensor(2 * torch.pi)
    else:
        c = (c1.pow(2) + c2.pow(2)) / h
    c = torch.exp(-0.5 * c)
    return c.mean(axis=0)


def kernel_gpu(x, h=None):
    c1 = x[:, 0].reshape(-1, 1) - x[:, 0]
    c2 = x[:, 1].reshape(-1, 1) - x[:, 1]
    if h is None:
        h = x.T.cov() * (x.shape[0] ** (-1 / 6))
        # h = x.T.cov() / x.shape[0]
        h = h.cpu().detach().numpy()
        h = np.linalg.inv(h)
        h = torch.tensor(h).cuda()
        c = h[0, 0] * c1.pow(2) + (h[0, 1] + h[1, 0]) * c1 * c2 + h[1, 1] * c2.pow(2)
        # c = torch.exp(-0.5 * c) * np.sqrt(h[0, 0] * h[1, 1] - h[0, 1] * h[1, 0])/ torch.tensor(2 * torch.pi)
    else:
        c = (c1.pow(2) + c2.pow(2)) / h
    c = torch.exp(-0.5 * c)
    return c.mean(axis=0)


def kernel_pred(x, y, h=None, return_cost=False):
    c1 = x[:, 0].reshape(-1, 1) - y[:, 0]
    c2 = x[:, 1].reshape(-1, 1) - y[:, 1]
    if return_cost or (h is not None):
        cost = c1.pow(2) + c2.pow(2)
    if h is None:
        h = x.T.cov() * (x.shape[0] ** -(1 / 6))
        # h = x.T.cov() / x.shape[0]
        h = h.detach().numpy()
        h = np.linalg.inv(h)
        c = h[0, 0] * c1.pow(2) + (h[0, 1] + h[1, 0]) * c1 * c2 + h[1, 1] * c2.pow(2)
        # c = torch.exp(-0.5 * c) *  np.sqrt(h[0, 0] * h[1, 1] - h[0, 1] * h[1, 0])/ torch.tensor(2 * torch.pi)
    else:
        c = cost / h
    c = torch.exp(-0.5 * c)
    if return_cost:
        return c.mean(axis=0), cost.T
    else:
        return c.mean(axis=0)


def kernel_pred_gpu(x, y, h=None, return_cost=False):
    c1 = x[:, 0].reshape(-1, 1) - y[:, 0]
    c2 = x[:, 1].reshape(-1, 1) - y[:, 1]
    if return_cost or (h is not None):
        cost = c1.pow(2) + c2.pow(2)
    if h is None:
        h = x.T.cov() * (x.shape[0] ** -(1 / 6))
        # h = x.T.cov() / x.shape[0]
        h = h.cpu().detach().numpy()
        h = np.linalg.inv(h)
        h = torch.tensor(h).cuda()
        c = h[0, 0] * c1.pow(2) + (h[0, 1] + h[1, 0]) * c1 * c2 + h[1, 1] * c2.pow(2)
        # c = torch.exp(-0.5 * c) *  np.sqrt(h[0, 0] * h[1, 1] - h[0, 1] * h[1, 0])/ torch.tensor(2 * torch.pi)
    else:
        c = cost / h
    c = torch.exp(-0.5 * c)
    if return_cost:
        return c.mean(axis=0), cost.T
    else:
        return c.mean(axis=0)


def kernel_weight(x, y, h=None):
    if h is None:
        h = y.T.cov() * (y.shape[0] ** -(1 / 6))
        # h = x.T.cov() / x.shape[0]
        h = h.detach().numpy()
    h = np.linalg.inv(h)
    c1 = x[:, 0].reshape(-1, 1) - y[:, 0]
    c2 = x[:, 1].reshape(-1, 1) - y[:, 1]
    c = h[0, 0] * c1.pow(2) + (h[0, 1] + h[1, 0]) * c1 * c2 + h[1, 1] * c2.pow(2)
    c = torch.exp(-0.5 * c)
    c = torch.diag(1 / c.sum(axis=1)) @ c
    return c


def kernel_weight_gpu(x, y, h=None):
    if h is None:
        h = y.T.cov() * (y.shape[0] ** -(1 / 6))
        # h = x.T.cov() / x.shape[0]
        h = h.cpu().detach().numpy()
    h = np.linalg.inv(h)
    h = torch.tensor(h).cuda()
    c1 = x[:, 0].reshape(-1, 1) - y[:, 0]
    c2 = x[:, 1].reshape(-1, 1) - y[:, 1]
    c = h[0, 0] * c1.pow(2) + (h[0, 1] + h[1, 0]) * c1 * c2 + h[1, 1] * c2.pow(2)
    c = torch.exp(-0.5 * c)
    c = torch.diag(1 / c.sum(axis=1)) @ c
    return c


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


def train_alg_mfc_ot(data, T, lr=0.001, n_layers=2,
                     n_sample=100, n_iter=128, nt_grid=100,
                     s1=1, s2=1,
                     h=None, k=5, lock_dist=0.01,
                     r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1,
                     reg=1, reg1=1, reg2=1,
                     track=False, **_):

    t_data = data.time.unique()
    t_data.sort()
    t_grid = np.unique(np.concatenate((t_data, np.linspace(0, T, nt_grid)), axis=None))
    t_grid.sort()
    nt_data = t_data.shape[0]
    nt_model = t_grid.shape[0]
    dirac = np.repeat(1, n_sample) / n_sample
    tmap = []
    start_loc = []
    end_loc = []
    for i in range(nt_data - 1):
        data0 = data[data.time == t_data[i]][['x', 'y']].sample(n_sample, replace=True).to_numpy()
        data1 = data[data.time == t_data[i + 1]][['x', 'y']].sample(n_sample, replace=True).to_numpy()
        costm = ot_num.compute_dist(data0, data1, dim=2, single=False)
        start_loc.append(data0)
        reg_list = (100 - reg) * np.exp(-np.arange(100)) + reg
        tmap_temp = ot_num.ot_unbalanced_log_stabilized(dirac, dirac, costm, reg, reg1, reg2, reg_list=reg_list)
        tmap.append(tmap_temp)
        tmap_norm = tmap_temp.copy()
        tmap_norm = np.diag(1 / tmap_norm.sum(axis=1)) @ tmap_norm
        # end_loc.append(tmap_norm @ data1)
        end_loc.append(data1[tmap_norm.argmax(axis=1)])

    model = NeuralNetwork(3, 2, 100, n_layers=n_layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2),
                                                                    torch.tensor(np.diag([s1, s2])).float())

    obj = []

    for n in range(n_iter):

        l = torch.tensor(0.)

        x = torch.from_numpy(start_loc[0])
        ind_check = 1

        check = True
        for t_ind in range(nt_model - 1):
            ti = t_grid[t_ind]
            tf = t_grid[t_ind + 1]
            dt = (tf - ti) / T
            inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
            v = model(inp)
            e = me.sample([n_sample])
            x = x + v * dt + np.sqrt(dt) * e
            l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
            phat = kernel(x, h=h)
            pvhat = kernel(v, h=h)
            l = l + r_ent_v * dt * pvhat.log().mean()
            if check:
                if tf == t_data[ind_check]:
                    x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample).to_numpy())
                    # x_support = torch.from_numpy(data.sample(1000)[['x', 'y']].to_numpy())
                    c1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
                    c2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
                    c = c1.pow(2) + c2.pow(2)
                    p = kernel_pred(x_check, x, h=h)
                    l = l - r_kl * p.log().mean()
                    c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                    ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                    l = l + r_lock * (torch.max(c_lowk.sqrt() - lock_dist, torch.tensor(0.))).sum(axis=1).mean()
                    l = l + r_lock * (torch.max(ctr_lowk.sqrt() - lock_dist, torch.tensor(0.))).sum(axis=0).mean()
                    x_ref = torch.tensor(end_loc[ind_check - 1])
                    l = l + (x - x_ref).pow(2).sum(axis=1).mean()
                    if tf == t_data[-1]:
                        check = False
                    ind_check += 1
                else:
                    l = l + r_ent * dt * (phat.log().mean())

        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        obj.append(float(l))
        if bool(l.isnan()):
            raise ArithmeticError('encountered nan at iteration ' + str(int(n)))
        if track:
            print('c = ', str(float(l)), ' at iteration ', str(int(n)))

    return {'model': model,
            'optimizer': optimizer,
            'cost': obj,
            'tmap': tmap,
            'x0': start_loc,
            'x1': end_loc,
            't_data': t_data}


def train_alg_mfc_ot_gpu(data, T, lr=0.001, n_layers=2,
                         n_sample=100, n_iter=128, nt_grid=100,
                         s1=1, s2=1,
                         h=None, k=5, lock_dist=0.01,
                         r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1,
                         reg=1, reg1=1, reg2=1,
                         track=False, **_):

    t_data = data.time.unique()
    t_data.sort()
    t_grid = np.unique(np.concatenate((t_data, np.linspace(0, T, nt_grid)), axis=None))
    t_grid.sort()
    nt_data = t_data.shape[0]
    nt_model = t_grid.shape[0]
    dirac = np.repeat(1, n_sample) / n_sample
    tmap = []
    start_loc = []
    end_loc = []
    for i in range(nt_data - 1):
        data0 = data[data.time == t_data[i]][['x', 'y']].sample(n_sample, replace=True).to_numpy()
        data1 = data[data.time == t_data[i + 1]][['x', 'y']].sample(n_sample, replace=True).to_numpy()
        costm = ot_num.compute_dist(data0, data1, dim=2, single=False)
        start_loc.append(data0)
        reg_list = (100 - reg) * np.exp(-np.arange(100)) + reg
        tmap_temp = ot_num.ot_unbalanced_log_stabilized(dirac, dirac, costm, reg, reg1, reg2, reg_list=reg_list)
        tmap.append(tmap_temp)
        tmap_norm = tmap_temp.copy()
        tmap_norm = np.diag(1 / tmap_norm.sum(axis=1)) @ tmap_norm
        # end_loc.append(tmap_norm @ data1)
        end_loc.append(data1[tmap_norm.argmax(axis=1)])

    model = NeuralNetwork(3, 2, 100, n_layers=n_layers).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2).cuda(),
                                                                    torch.tensor(np.diag([s1, s2])).cuda().float())

    obj = []

    for n in range(n_iter):

        l = torch.tensor(0.).cuda()

        x = torch.from_numpy(start_loc[0]).cuda()
        ind_check = 1

        check = True
        for t_ind in range(nt_model - 1):
            ti = t_grid[t_ind]
            tf = t_grid[t_ind + 1]
            dt = (tf - ti) / T
            inp = torch.cat([x, ti * torch.ones(n_sample, 1).cuda() / T], dim=1)
            v = model(inp)
            e = me.sample([n_sample])
            x = x + v * dt + np.sqrt(dt) * e
            l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
            phat = kernel_gpu(x, h=h)
            pvhat = kernel_gpu(v, h=h)
            l = l + r_ent_v * dt * pvhat.log().mean()
            if check:
                if tf == t_data[ind_check]:
                    x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample).to_numpy()).cuda()
                    # x_support = torch.from_numpy(data.sample(1000)[['x', 'y']].to_numpy())
                    c1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
                    c2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
                    c = c1.pow(2) + c2.pow(2)
                    p = kernel_pred_gpu(x_check, x, h=h)
                    l = l - r_kl * p.log().mean()
                    c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                    ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                    l = l + r_lock * (torch.max(c_lowk.sqrt() - lock_dist, torch.tensor(0.).cuda())).sum(axis=1).mean()
                    l = l + r_lock * (torch.max(ctr_lowk.sqrt() - lock_dist, torch.tensor(0.).cuda())).sum(axis=0).mean()
                    x_ref = torch.tensor(end_loc[ind_check - 1]).cuda()
                    l = l + (x - x_ref).pow(2).sum(axis=1).mean()
                    if tf == t_data[-1]:
                        check = False
                    ind_check += 1
                else:
                    l = l + r_ent * dt * (phat.log().mean())

        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        obj.append(float(l))
        if bool(l.isnan()):
            raise ArithmeticError('encountered nan at iteration ' + str(int(n)))
        if track:
            print('c = ', str(float(l)), ' at iteration ', str(int(n)))

    return {'model': model,
            'optimizer': optimizer,
            'cost': obj,
            'tmap': tmap,
            'x0': start_loc,
            'x1': end_loc,
            't_data': t_data}
    pass


def train_alg_mfc_force(data, T, lr=0.001, n_layers=2,
                        n_sample=100, n_iter=128, nt_grid=100,
                        s1=1, s2=1,
                        h=None,
                        r_v=0.01, r_ent=0.1, r_kl=1,
                        track=False, **_):

    t_data = data.time.unique()
    t_data.sort()
    t_grid = np.unique(np.concatenate((t_data, np.linspace(0, T, nt_grid)), axis=None))
    t_grid.sort()
    nt = t_grid.shape[0]

    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2),
                                                                    torch.tensor(np.diag([s1, s2])).float())

    model = NeuralNetwork(3, 2, 100, n_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

    x0 = data[data.time == 0]

    x_tensor = []

    for t in t_data:
        x_tensor.append(torch.from_numpy(data[data.time == t][['x','y']].to_numpy()))

    obj = []

    for n in range(n_iter):

        x = torch.from_numpy(x0.sample(n_sample, replace=True)[['x','y']].to_numpy())
        ind_check = 1
        check = True
        l = torch.tensor(0.)
        for t_ind in range(nt - 1):
            ti = t_grid[t_ind]
            tf = t_grid[t_ind + 1]
            dt = (tf - ti) / T
            inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
            e = me.sample([n_sample])
            if check:
                tc = t_data[ind_check]
                gamma = (tf - ti) / (tc - ti)
                if gamma < 0 or gamma > 1:
                    raise ArithmeticError('encountered wrong gamma')
                x_check = x_tensor[ind_check]
                d1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
                d2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
                c = d1.pow(2) + d2.pow(2)
                ind_close = c.argmin(axis=1)
                dest_ind = np.zeros(n_sample)
                for i in range(n_sample):
                    if np.random.uniform() > (1 - gamma):
                        dest_ind[i] = np.random.choice(np.arange(x_check.shape[0]))
                    else:
                        dest_ind[i] = ind_close[i]
                x_pred = x_check[dest_ind]
                v_pred = (x_pred - x) / dt
                if tf == t_data[ind_check]:
                    if tf == t_data[-1]:
                        check = False
                    ind_check += 1
                    v = v_pred
                    x = x + v * dt + np.sqrt(dt) * e
                    p = kernel_pred(x_check, x, h=h)
                    l = l - r_kl * p.log().mean()
                else:
                    v = model(inp) * (1 - gamma) + v_pred * gamma
                    x = x + v * dt + np.sqrt(dt) * e
                    phat = kernel(x, h=h)
                    l = l + r_ent * dt * (phat.log().mean())
                l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
            else:
                v = model(inp)
                l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
                x = x + v * dt + np.sqrt(dt) * e
                # x = x + v * dt
                l = l + r_ent * dt * phat.log().mean()

        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        obj.append(float(l))
        if bool(l.isnan()):
            raise ArithmeticError('encountered nan at iteration ' + str(int(n)))
        if track:
            print('c = ', str(float(l)), ' at iteration ', str(int(n)))

    return {'model': model,
            'optimizer': optimizer,
            'cost': obj}


def train_alg_mfc_force_gpu(data, T, lr=0.001, n_layers=2,
                            n_sample=100, n_iter=128, nt_grid=100,
                            s1=1, s2=1,
                            h=None,
                            r_v=0.01, r_ent=0.1, r_kl=1,
                            track=False, **_):

    t_data = data.time.unique()
    t_data.sort()
    t_grid = np.unique(np.concatenate((t_data, np.linspace(0, T, nt_grid)), axis=None))
    t_grid.sort()
    nt = t_grid.shape[0]

    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2).cuda(),
                                                                    torch.tensor(np.diag([s1, s2])).cuda().float())

    model = NeuralNetwork(3, 2, 100, n_layers=n_layers).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

    x0 = data[data.time == 0]

    x_tensor = []

    for t in t_data:
        x_tensor.append(torch.from_numpy(data[data.time == t][['x','y']].to_numpy()).cuda())

    obj = []

    for n in range(n_iter):

        x = torch.from_numpy(x0.sample(n_sample, replace=True)[['x','y']].to_numpy()).cuda()
        ind_check = 1
        check = True
        l = torch.tensor(0.)
        for t_ind in range(nt - 1):
            ti = t_grid[t_ind]
            tf = t_grid[t_ind + 1]
            dt = (tf - ti) / T
            inp = torch.cat([x, ti * torch.ones(n_sample, 1).cuda() / T], dim=1)
            e = me.sample([n_sample])
            if check:
                tc = t_data[ind_check]
                gamma = (tf - ti) / (tc - ti)
                if gamma < 0 or gamma > 1:
                    raise ArithmeticError('encountered wrong gamma')
                x_check = x_tensor[ind_check]
                d1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
                d2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
                c = d1.pow(2) + d2.pow(2)
                ind_close = c.argmin(axis=1)
                dest_ind = np.zeros(n_sample)
                for i in range(n_sample):
                    if np.random.uniform() > (1 - gamma):
                        dest_ind[i] = np.random.choice(np.arange(x_check.shape[0]))
                    else:
                        dest_ind[i] = ind_close[i]
                x_pred = x_check[dest_ind]
                v_pred = (x_pred - x) / dt
                if tf == t_data[ind_check]:
                    if tf == t_data[-1]:
                        check = False
                    ind_check += 1
                    v = v_pred
                    x = x + v * dt + np.sqrt(dt) * e
                    p = kernel_pred_gpu(x_check, x, h=h)
                    l = l - r_kl * p.log().mean()
                else:
                    v = model(inp) * (1 - gamma) + v_pred * gamma
                    x = x + v * dt + np.sqrt(dt) * e
                    phat = kernel_gpu(x, h=h)
                    l = l + r_ent * dt * (phat.log().mean())
                l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
            else:
                v = model(inp)
                l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
                x = x + v * dt + np.sqrt(dt) * e
                # x = x + v * dt
                l = l + r_ent * dt * phat.log().mean()

        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        obj.append(float(l))
        if bool(l.isnan()):
            raise ArithmeticError('encountered nan at iteration ' + str(int(n)))
        if track:
            print('c = ', str(float(l)), ' at iteration ', str(int(n)))

    return {'model': model,
            'optimizer': optimizer,
            'cost': obj}


def train_alg_mfc_soft(data, T, lr=0.001, n_layers=2,
                       n_sample=100, n_iter=128, nt_grid=100,
                       s1=1, s2=1,
                       h=None, k=5, lock_dist=0.01,
                       r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock1=1, r_lock2=1,
                       track=False, **_):

    t_data = data.time.unique()
    t_data.sort()
    t_grid = np.unique(np.concatenate((t_data, np.linspace(0, T, nt_grid)), axis=None))
    t_grid.sort()
    nt = t_grid.shape[0]

    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2),
                                                                    torch.tensor(np.diag([s1, s2])).float())

    model = NeuralNetwork(3, 2, 128, n_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

    x0 = data[data.time == 0]

    # x_tensor = []
    # for t in t_data:
    #     x_tensor.append(torch.from_numpy(data[data.time == t][['x','y']].to_numpy()))

    obj = []

    for n in range(n_iter):

        x = torch.from_numpy(x0.sample(n_sample, replace=True)[['x','y']].to_numpy())
        ind_check = 1
        check = True
        l = torch.tensor(0.)
        for t_ind in range(nt - 1):
            ti = t_grid[t_ind]
            tf = t_grid[t_ind + 1]
            dt = (tf - ti) / T
            inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
            v = model(inp)
            e = me.sample([n_sample])
            x = x + v * dt + np.sqrt(dt) * e
            l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
            phat = kernel(x, h=h)
            # pvhat = kernel(v, h=h)
            # l = l + r_ent_v * dt * (pvhat.log().mean())
            if check:
                if tf == t_data[ind_check]:
                    x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample).to_numpy())
                    # x_support = torch.from_numpy(data.sample(1000)[['x', 'y']].to_numpy())
                    c1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
                    c2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
                    c = c1.pow(2) + c2.pow(2)
                    # prop_in = torch.diag(1 / c.sum(axis=1)) @ c
                    p = kernel_pred(x_check, x, h=h)
                    l = l - r_kl * p.log().mean()
                    c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                    ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                    l = l + r_lock1 * (torch.max(c_lowk.sqrt() - lock_dist, torch.tensor(0.))).sum(axis=1).mean()
                    l = l + r_lock2 * (torch.max(ctr_lowk.sqrt() - lock_dist, torch.tensor(0.))).sum(axis=0).mean()
                    if tf == t_data[-1]:
                        check = False
                    ind_check += 1
                else:
                    l = l + r_ent * dt * (phat.log().mean())
                # l = l + r_ent * dt * (phat.log().mean())

        if bool(l.isnan()):
            raise ArithmeticError('encountered nan at iteration ' + str(int(n)))
        if track:
            print('c = ', str(float(l)), ' at iteration ', str(int(n)))

        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        obj.append(float(l))

    return {'model': model,
            'optimizer': optimizer,
            'cost': obj}


def train_alg_mfc_soft_gpu(data, T, lr=0.001, n_layers=2,
                           n_sample=100, n_iter=128, nt_grid=100,
                           s1=1, s2=1,
                           h=None, k=5, lock_dist=0.01,
                           r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1,
                           track=False, **_):

    t_data = data.time.unique()
    t_data.sort()
    t_grid = np.unique(np.concatenate((t_data, np.linspace(0, T, nt_grid)), axis=None))
    t_grid.sort()
    nt = t_grid.shape[0]

    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2).cuda(),
                                                                    torch.tensor(np.diag([s1, s2])).cuda().float())

    model = NeuralNetwork(3, 2, 128, n_layers=n_layers).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

    x0 = data[data.time == 0]

    # x_tensor = []
    # for t in t_data:
    #     x_tensor.append(torch.from_numpy(data[data.time == t][['x','y']].to_numpy()))

    obj = []

    for n in range(n_iter):

        x = torch.from_numpy(x0.sample(n_sample, replace=True)[['x','y']].to_numpy()).cuda()
        ind_check = 1
        check = True
        l = torch.tensor(0.).cuda()
        for t_ind in range(nt - 1):
            ti = t_grid[t_ind]
            tf = t_grid[t_ind + 1]
            dt = (tf - ti) / T
            inp = torch.cat([x, ti * torch.ones(n_sample, 1).cuda() / T], dim=1)
            v = model(inp)
            e = me.sample([n_sample])
            x = x + v * dt + np.sqrt(dt) * e
            l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
            phat = kernel_gpu(x, h=h)
            # pvhat = kernel_gpu(v, h=h)
            # l = l + r_ent_v * dt * (pvhat.log().mean())
            if check:
                if tf == t_data[ind_check]:
                    x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample).to_numpy()).cuda()
                    # x_support = torch.from_numpy(data.sample(1000)[['x', 'y']].to_numpy())
                    c1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
                    c2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
                    c = c1.pow(2) + c2.pow(2)
                    # prop_in = torch.diag(1 / c.sum(axis=1)) @ c
                    p = kernel_pred_gpu(x_check, x, h=h)
                    l = l - r_kl * p.log().mean()
                    c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                    ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                    l = l + r_lock * (torch.max(c_lowk.sqrt() - lock_dist, torch.tensor(0.).cuda())).sum(axis=1).mean()
                    l = l + r_lock * (torch.max(ctr_lowk.sqrt() - lock_dist, torch.tensor(0.).cuda())).sum(axis=0).mean()
                    if tf == t_data[-1]:
                        check = False
                    ind_check += 1
                else:
                    l = l + r_ent * dt * (phat.log().mean())
                # l = l + r_ent * dt * (phat.log().mean())

        if bool(l.isnan()):
            raise ArithmeticError('encountered nan at iteration ' + str(int(n)))
        if track:
            print('c = ', str(float(l)), ' at iteration ', str(int(n)))

        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        obj.append(float(l))

    return {'model': model,
            'optimizer': optimizer,
            'cost': obj}


def train_alg_mfc_fbsde(data, T, lr=0.001, n_layers=2,
                        n_sample=100, n_iter=128, nt_grid=100, fb_iter=10,
                        s1=1, s2=1,
                        h=1, k=5, lock_dist=0.01, use_score=False,
                        r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1, beta=1,
                        track=False, **_):

    t_data = data.time.unique()
    t_data.sort()
    t_grid = np.unique(np.concatenate((t_data, np.linspace(0, T, nt_grid)), axis=None))
    t_grid.sort()
    nt = t_grid.shape[0]

    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2),
                                                                    torch.tensor(np.diag([s1, s2])).float())

    model_f = NeuralNetwork(3, 2, 128, n_layers=n_layers)
    model_b = NeuralNetwork(3, 2, 128, n_layers=n_layers)
    optimizer_f = torch.optim.Adam(model_f.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

    x0 = data[data.time == 0]
    x1 = data[data.time == data.time.max()]

    obj_f = []
    obj_b = []

    x_target_track = []
    x_start = x0.sample(n_sample, replace=True)[['x','y']].to_numpy()

    for i_fb in range(fb_iter):

        obj = []

        if i_fb > 0:
            x_target = x_target_track[-1]

        if i_fb % 2 == 0: # forward modeling

            for n in range(n_iter):

                x = torch.from_numpy(x_start)
                x_start = x.detach().numpy()
                ind_check = 1
                check = True
                l = torch.tensor(0.)
                for t_ind in range(nt - 1):
                    ti = t_grid[t_ind]
                    tf = t_grid[t_ind + 1]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
                    v = model_f(inp)
                    e = me.sample([n_sample])
                    x = x + v * dt + np.sqrt(dt) * e
                    l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
                    # pvhat = kernel(v, h=h)
                    # l = l + r_ent_v * dt * (pvhat.log().mean())
                    if check:
                        if tf == t_data[ind_check]:
                            x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample, replace=True).to_numpy())
                            p, c = kernel_pred(x_check, x, h=h, return_cost=True)
                            l = l - r_kl * p.log().mean()
                            c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                            ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                            l = l + r_lock * (torch.max(c_lowk - lock_dist, torch.tensor(0.))).sum(axis=1).mean()
                            l = l + r_lock * (torch.max(ctr_lowk - lock_dist, torch.tensor(0.))).sum(axis=0).mean()
                            if tf == t_data[-1]:
                                check = False
                            ind_check += 1
                        else:
                            phat = kernel(x, h=h)
                            l = l + r_ent * dt * (phat.log().mean())
                            pass
                    if i_fb > 0:
                        x_ref = torch.tensor(x_target[t_ind + 1])
                        l = l + beta * (x - x_ref).pow(2).sum(axis=1).mean()

                if bool(l.isnan()):
                    raise ArithmeticError('encountered nan at forward iteration ' + str(i_fb) + ' iteration ' + str(int(n)))
                if track:
                    print('c = ', str(float(l)), ' at forward iteration ' + str(i_fb) + ' iteration ' + str(int(n)))

                optimizer_f.zero_grad()
                l.backward()
                optimizer_f.step()
                obj.append(float(l))

            if use_score:
                x = torch.tensor(x1.sample(n_sample, replace=True)[['x','y']].to_numpy())
                x_start = x.detach().numpy()
                x_track_temp = []
                x_track_temp.append(x.detach().numpy())
                for t_ind in range(nt - 1):
                    ti = t_grid[-(t_ind + 1)]
                    tf = t_grid[-(t_ind + 2)]
                    dt = (ti - tf) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
                    v = model_f(inp)
                    e = me.sample([n_sample])
                    score = score_est(x, x, h=h)
                    x = x + (v - s1 * score) * dt + np.sqrt(dt) * e
                    x_track_temp.append(x.detach().numpy())
                x_target_track.append(x_track_temp)
            else:
                x = torch.tensor(x0.sample(n_sample, replace=True)[['x','y']].to_numpy())
                x_track_temp = []
                x_track_temp.append(x.detach().numpy())
                for t_ind in range(nt - 1):
                    ti = t_grid[t_ind]
                    tf = t_grid[t_ind + 1]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
                    v = model_f(inp)
                    e = me.sample([n_sample])
                    x = x + v * dt + np.sqrt(dt) * e
                    x_track_temp.append(x.detach().numpy())
                x_target_track.append(x_track_temp[::-1])
                x_start = x.detach().numpy()

            obj_f.append(obj)

        else: # backward modeling

            for n in range(n_iter):

                x = torch.from_numpy(x_start)
                x_start = x.detach().numpy()
                ind_check = 1
                check = True
                l = torch.tensor(0.)
                for t_ind in range(nt - 1):
                    ti = T - t_grid[-(t_ind + 1)]
                    tf = T - t_grid[-(t_ind + 2)]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
                    v = model_b(inp)
                    e = me.sample([n_sample])
                    x = x + v * dt + np.sqrt(dt) * e
                    l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
                    # pvhat = kernel(v, h=h)
                    # l = l + r_ent_v * dt * (pvhat.log().mean())
                    if check:
                        if tf == t_data[-(ind_check + 1)]:
                            x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample, replace=True).to_numpy())
                            p, c = kernel_pred(x_check, x, h=h, return_cost=True)
                            l = l - r_kl * p.log().mean()
                            c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                            ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                            l = l + r_lock * (torch.max(c_lowk - lock_dist, torch.tensor(0.))).sum(axis=1).mean()
                            l = l + r_lock * (torch.max(ctr_lowk - lock_dist, torch.tensor(0.))).sum(axis=0).mean()
                            if tf == t_data[0]:
                                check = False
                            ind_check += 1
                        else:
                            phat = kernel(x, h=h)
                            l = l + r_ent * dt * (phat.log().mean())
                            pass
                    if i_fb > 0:
                        x_ref = torch.tensor(x_target[t_ind + 1])
                        l = l + beta * (x - x_ref).pow(2).sum(axis=1).mean()

                if bool(l.isnan()):
                    raise ArithmeticError('encountered nan at backward iteration ' + str(i_fb) + ' iteration ' + str(int(n)))
                if track:
                    print('c = ', str(float(l)), ' at backward iteration ' + str(i_fb) + ' iteration ' + str(int(n)))

                optimizer_b.zero_grad()
                l.backward()
                optimizer_b.step()
                obj.append(float(l))

            if use_score:
                x = torch.tensor(x0.sample(n_sample, replace=True)[['x','y']].to_numpy())
                x_start = x.detach().numpy()
                x_track_temp = []
                x_track_temp.append(x.detach().numpy())
                for t_ind in range(nt - 1):
                    ti = t_grid[t_ind]
                    tf = t_grid[t_ind + 1]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, (T - ti) * torch.ones(n_sample, 1) / T], dim=1)
                    v = model_b(inp)
                    e = me.sample([n_sample])
                    score = score_est(x, x, h=h)
                    x = x + (v + s1 * score) * dt + np.sqrt(dt) * e
                    x_track_temp.append(x.detach().numpy())
                x_target_track.append(x_track_temp)
            else:
                x = torch.tensor(x1.sample(n_sample, replace=True)[['x','y']].to_numpy())
                x_track_temp = []
                x_track_temp.append(x.detach().numpy())
                for t_ind in range(nt - 1):
                    ti = T - t_grid[-(t_ind + 1)]
                    tf = T - t_grid[-(t_ind + 2)]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
                    v = model_b(inp)
                    e = me.sample([n_sample])
                    x = x + v * dt + np.sqrt(dt) * e
                    x_track_temp.append(x.detach().numpy())
                x_target_track.append(x_track_temp[::-1])
                x_start = x.detach().numpy()

            obj_b.append(obj)

    return {'model_f': model_f,
            'model_b': model_b,
            'optimizer_f': optimizer_f,
            'optimizer_b': optimizer_b,
            'cost_f': obj_f,
            'cost_b': obj_b}


def train_alg_mfc_fbsde_gpu(data, T, lr=0.001, n_layers=2,
                            n_sample=100, n_iter=128, nt_grid=100, fb_iter=10,
                            s1=1, s2=1,
                            h=1, k=5, lock_dist=0.01, use_score=False,
                            r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1, beta=1,
                            track=False, **_):

    t_data = data.time.unique()
    t_data.sort()
    t_grid = np.unique(np.concatenate((t_data, np.linspace(0, T, nt_grid)), axis=None))
    t_grid.sort()
    nt = t_grid.shape[0]

    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2).cuda(),
                                                                    torch.tensor(np.diag([s1, s2])).cuda().float())

    model_f = NeuralNetwork(3, 2, 128, n_layers=n_layers).cuda()
    model_b = NeuralNetwork(3, 2, 128, n_layers=n_layers).cuda()
    optimizer_f = torch.optim.Adam(model_f.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

    x0 = data[data.time == 0]
    x1 = data[data.time == data.time.max()]

    obj_f = []
    obj_b = []

    x_target_track = []
    x_start = x0.sample(n_sample, replace=True)[['x','y']].to_numpy()

    for i_fb in range(fb_iter):

        obj = []

        if i_fb > 0:
            x_target = x_target_track[-1]

        if i_fb % 2 == 0: # forward modeling

            for n in range(n_iter):

                x = torch.from_numpy(x_start).cuda()
                x_start = x.cpu().detach().numpy()
                ind_check = 1
                check = True
                l = torch.tensor(0.).cuda()
                for t_ind in range(nt - 1):
                    ti = t_grid[t_ind]
                    tf = t_grid[t_ind + 1]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1).cuda() / T], dim=1)
                    v = model_f(inp)
                    e = me.sample([n_sample])
                    x = x + v * dt + np.sqrt(dt) * e
                    l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
                    # pvhat = kernel_gpu(v, h=h)
                    # l = l + r_ent_v * dt * (pvhat.log().mean())
                    if check:
                        if tf == t_data[ind_check]:
                            x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample, replace=True).to_numpy()).cuda()
                            p, c = kernel_pred_gpu(x_check, x, h=h, return_cost=True)
                            l = l - r_kl * p.log().mean()
                            c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                            ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                            l = l + r_lock * (torch.max(c_lowk - lock_dist, torch.tensor(0.).cuda())).sum(axis=1).mean()
                            l = l + r_lock * (torch.max(ctr_lowk - lock_dist, torch.tensor(0.).cuda())).sum(axis=0).mean()
                            if tf == t_data[-1]:
                                check = False
                            ind_check += 1
                        else:
                            phat = kernel_gpu(x, h=h)
                            l = l + r_ent * dt * (phat.log().mean())
                            pass
                    if i_fb > 0:
                        x_ref = torch.tensor(x_target[t_ind + 1]).cuda()
                        l = l + beta * (x - x_ref).pow(2).sum(axis=1).mean()

                if bool(l.isnan()):
                    raise ArithmeticError('encountered nan at forward iteration ' + str(i_fb) + ' iteration ' + str(int(n)))
                if track:
                    print('c = ', str(float(l)), ' at forward iteration ' + str(i_fb) + ' iteration ' + str(int(n)))

                optimizer_f.zero_grad()
                l.backward()
                optimizer_f.step()
                obj.append(float(l))

            if use_score:
                x = torch.tensor(x1.sample(n_sample, replace=True)[['x','y']].to_numpy()).cuda()
                x_start = x.cpu().detach().numpy()
                x_track_temp = []
                x_track_temp.append(x.cpu().detach().numpy())
                for t_ind in range(nt - 1):
                    ti = t_grid[-(t_ind + 1)]
                    tf = t_grid[-(t_ind + 2)]
                    dt = (ti - tf) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1).cuda() / T], dim=1)
                    v = model_f(inp)
                    e = me.sample([n_sample])
                    score = score_est_gpu(x, x, h=h)
                    x = x + (v - s1 * score) * dt + np.sqrt(dt) * e
                    x_track_temp.append(x.cpu().detach().numpy())
                x_target_track.append(x_track_temp)
            else:
                x = torch.tensor(x0.sample(n_sample, replace=True)[['x','y']].to_numpy()).cuda()
                x_track_temp = []
                x_track_temp.append(x.cpu().detach().numpy())
                for t_ind in range(nt - 1):
                    ti = t_grid[t_ind]
                    tf = t_grid[t_ind + 1]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1).cuda() / T], dim=1)
                    v = model_f(inp)
                    e = me.sample([n_sample])
                    x = x + v * dt + np.sqrt(dt) * e
                    x_track_temp.append(x.cpu().detach().numpy())
                x_target_track.append(x_track_temp[::-1])
                x_start = x.cpu().detach().numpy()

            obj_f.append(obj)

        else: # backward modeling

            for n in range(n_iter):

                x = torch.from_numpy(x_start).cuda()
                x_start = x.cpu().detach().numpy()
                ind_check = 1
                check = True
                l = torch.tensor(0.).cuda()
                for t_ind in range(nt - 1):
                    ti = T - t_grid[-(t_ind + 1)]
                    tf = T - t_grid[-(t_ind + 2)]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1).cuda() / T], dim=1)
                    v = model_b(inp)
                    e = me.sample([n_sample])
                    x = x + v * dt + np.sqrt(dt) * e
                    l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
                    # pvhat = kernel_gpu(v, h=h)
                    # l = l + r_ent_v * dt * (pvhat.log().mean())
                    if check:
                        if tf == t_data[-(ind_check + 1)]:
                            x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample, replace=True).to_numpy()).cuda()
                            p, c = kernel_pred_gpu(x_check, x, h=h, return_cost=True)
                            l = l - r_kl * p.log().mean()
                            c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                            ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                            l = l + r_lock * (torch.max(c_lowk - lock_dist, torch.tensor(0.).cuda())).sum(axis=1).mean()
                            l = l + r_lock * (torch.max(ctr_lowk - lock_dist, torch.tensor(0.).cuda())).sum(axis=0).mean()
                            if tf == t_data[0]:
                                check = False
                            ind_check += 1
                        else:
                            phat = kernel_gpu(x, h=h)
                            l = l + r_ent * dt * (phat.log().mean())
                            pass
                    if i_fb > 0:
                        x_ref = torch.tensor(x_target[t_ind + 1]).cuda()
                        l = l + beta * (x - x_ref).pow(2).sum(axis=1).mean()

                if bool(l.isnan()):
                    raise ArithmeticError('encountered nan at backward iteration ' + str(i_fb) + ' iteration ' + str(int(n)))
                if track:
                    print('c = ', str(float(l)), ' at backward iteration ' + str(i_fb) + ' iteration ' + str(int(n)))

                optimizer_b.zero_grad()
                l.backward()
                optimizer_b.step()
                obj.append(float(l))

            if use_score:
                x = torch.tensor(x0.sample(n_sample, replace=True)[['x','y']].to_numpy()).cuda()
                x_start = x.cpu().detach().numpy()
                x_track_temp = []
                x_track_temp.append(x.cpu().detach().numpy())
                for t_ind in range(nt - 1):
                    ti = t_grid[t_ind]
                    tf = t_grid[t_ind + 1]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, (T - ti) * torch.ones(n_sample, 1).cuda() / T], dim=1)
                    v = model_b(inp)
                    e = me.sample([n_sample])
                    score = score_est_gpu(x, x, h=h)
                    x = x + (v + s1 * score) * dt + np.sqrt(dt) * e
                    x_track_temp.append(x.cpu().detach().numpy())
                x_target_track.append(x_track_temp)
            else:
                x = torch.tensor(x1.sample(n_sample, replace=True)[['x','y']].to_numpy()).cuda()
                x_track_temp = []
                x_track_temp.append(x.cpu().detach().numpy())
                for t_ind in range(nt - 1):
                    ti = T - t_grid[-(t_ind + 1)]
                    tf = T - t_grid[-(t_ind + 2)]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1).cuda() / T], dim=1)
                    v = model_b(inp)
                    e = me.sample([n_sample])
                    x = x + v * dt + np.sqrt(dt) * e
                    x_track_temp.append(x.cpu().detach().numpy())
                x_target_track.append(x_track_temp[::-1])
                x_start = x.cpu().detach().numpy()

            obj_b.append(obj)

    return {'model_f': model_f,
            'model_b': model_b,
            'optimizer_f': optimizer_f,
            'optimizer_b': optimizer_b,
            'cost_f': obj_f,
            'cost_b': obj_b}


def train_alg_mfc_soft_seg(data, T, lr=0.001, n_layers=2,
                           n_sample=100, n_iter=128, nt_grid=100, n_seg=5,
                           s1=1, s2=1,
                           h=None, k=5, lock_dist=0.01,
                           r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1,
                           track=False, **_):

    t_data = data.time.unique()
    t_data.sort()
    if n_seg is None:
        nt_batch = 1
    else:
        nt_batch = max(1, np.ceil((len(t_data) - 1) / n_seg).astype(int))

    t_data_seg = np.empty(n_seg, dtype=object)
    for ns in range(n_seg):
        t_data_seg[ns] = t_data[(ns * nt_batch):((ns + 1) * nt_batch + 1)]

    boundaries = np.zeros((n_seg, 2))
    boundaries[:, 0] = np.array([ts[0] for ts in t_data_seg])
    boundaries[:, 1] = np.array([ts[-1] for ts in t_data_seg])

    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2),
                                                                    torch.tensor(np.diag([s1, s2])).float())

    model = []
    optimizer = []
    obj = []

    for i in range(n_seg):
        model.append(NeuralNetwork(3, 2, 64, n_layers=n_layers))
        optimizer.append(torch.optim.Adam(model[i].parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5))
        obj.append([])

    for ns in range(n_seg):

        tsub = t_data_seg[ns]
        t_grid = np.concatenate((tsub, np.linspace(tsub[0], tsub[-1], nt_grid + 1)), axis=None)
        t_grid.sort()
        nt = len(t_grid)

        x0 = data[data.time == tsub[0]]
        x_tensor = []

        for t in tsub:
            x_tensor.append(torch.from_numpy(data[data.time == t][['x','y']].to_numpy()))

        for n in range(n_iter):

            x = torch.from_numpy(x0.sample(n_sample, replace=True)[['x','y']].to_numpy())
            ind_check = 1
            check = True
            l = torch.tensor(0.)
            for t_ind in range(nt - 1):
                ti = t_grid[t_ind]
                tf = t_grid[t_ind + 1]
                dt = (tf - ti) / T
                inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
                v = model[ns](inp)
                e = me.sample([n_sample])
                x = x + v * dt + np.sqrt(dt) * e
                l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
                pvhat = kernel(v, h=h)
                l = l + r_ent_v * dt * (pvhat.log().mean())
                if check:
                    if tf == tsub[ind_check]:
                        x_check = x_tensor[ind_check]
                        # if bool((p.log().abs() > 100).any()):
                        #     raise ArithmeticError('encountered nan at t = ' + str(ti) + ' iteration ' + str(int(n)))
                        c1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
                        c2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
                        c = c1.pow(2) + c2.pow(2)
                        # prop_in = torch.diag(1 / c.sum(axis=1)) @ c
                        p = kernel_pred(x_check, x, h=h)
                        l = l - r_kl * p.log().mean()
                        c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                        ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                        l = l + r_lock * (torch.max(c_lowk.sqrt() - lock_dist, torch.tensor(0.))).sum(axis=1).mean()
                        l = l + r_lock * (torch.max(ctr_lowk.sqrt() - lock_dist, torch.tensor(0.))).sum(axis=0).mean()
                        if tf == tsub[-1]:
                            check = False
                        ind_check += 1
                    else:
                        phat = kernel(x, h=h)
                        l = l + r_ent * dt * (phat.log().mean())

            if bool(l.isnan()):
                raise ArithmeticError('encountered nan at iteration ' + str(int(n)) + ' batch ' + str(int(ns)))
            if track:
                print('c = ', str(float(l)), ' at iteration ', str(int(n)) + ' batch ' + str(int(ns)))

            optimizer[ns].zero_grad()
            l.backward()
            optimizer[ns].step()
            obj[ns].append(float(l))

    return {'model': model,
            'optimizer': optimizer,
            'cost': obj,
            'bound': boundaries}


def train_alg_mfc_soft_seg_gpu(data, T, lr=0.001, n_layers=2,
                               n_sample=100, n_iter=128, nt_grid=100, n_seg=5,
                               s1=1, s2=1,
                               h=None, k=5, lock_dist=0.01,
                               r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1,
                               track=False, **_):

    t_data = data.time.unique()
    t_data.sort()
    if n_seg is None:
        nt_batch = 1
    else:
        nt_batch = max(1, np.ceil((len(t_data) - 1) / n_seg).astype(int))

    t_data_seg = np.empty(n_seg, dtype=object)
    for ns in range(n_seg):
        t_data_seg[ns] = t_data[(ns * nt_batch):((ns + 1) * nt_batch + 1)]

    boundaries = np.zeros((n_seg, 2))
    boundaries[:, 0] = np.array([ts[0] for ts in t_data_seg])
    boundaries[:, 1] = np.array([ts[-1] for ts in t_data_seg])

    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2).cuda(),
                                                                    torch.tensor(np.diag([s1, s2])).cuda().float())

    model = []
    optimizer = []
    obj = []

    for i in range(n_seg):
        model.append(NeuralNetwork(3, 2, 64, n_layers=n_layers).cuda())
        optimizer.append(torch.optim.Adam(model[i].parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5))
        obj.append([])

    for ns in range(n_seg):

        tsub = t_data_seg[ns]
        t_grid = np.concatenate((tsub, np.linspace(tsub[0], tsub[-1], nt_grid + 1)), axis=None)
        t_grid.sort()
        nt = len(t_grid)

        x0 = data[data.time == tsub[0]]
        x_tensor = []

        for t in tsub:
            x_tensor.append(torch.from_numpy(data[data.time == t][['x','y']].to_numpy()).cuda())

        for n in range(n_iter):

            x = torch.from_numpy(x0.sample(n_sample, replace=True)[['x','y']].to_numpy()).cuda()
            ind_check = 1
            check = True
            l = torch.tensor(0.).cuda()
            for t_ind in range(nt - 1):
                ti = t_grid[t_ind]
                tf = t_grid[t_ind + 1]
                dt = (tf - ti) / T
                inp = torch.cat([x, ti * torch.ones(n_sample, 1).cuda() / T], dim=1)
                v = model[ns](inp)
                e = me.sample([n_sample])
                x = x + v * dt + np.sqrt(dt) * e
                l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
                pvhat = kernel_gpu(v, h=h)
                l = l + r_ent_v * dt * (pvhat.log().mean())
                if check:
                    if tf == tsub[ind_check]:
                        x_check = x_tensor[ind_check]
                        # if bool((p.log().abs() > 100).any()):
                        #     raise ArithmeticError('encountered nan at t = ' + str(ti) + ' iteration ' + str(int(n)))
                        c1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
                        c2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
                        c = c1.pow(2) + c2.pow(2)
                        # prop_in = torch.diag(1 / c.sum(axis=1)) @ c
                        p = kernel_pred_gpu(x_check, x, h=h)
                        l = l - r_kl * p.log().mean()
                        c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                        ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                        l = l + r_lock * (torch.max(c_lowk.sqrt() - lock_dist, torch.tensor(0.).cuda())).sum(axis=1).mean()
                        l = l + r_lock * (torch.max(ctr_lowk.sqrt() - lock_dist, torch.tensor(0.).cuda())).sum(axis=0).mean()
                        if tf == tsub[-1]:
                            check = False
                        ind_check += 1
                    else:
                        phat = kernel_gpu(x, h=h)
                        l = l + r_ent * dt * (phat.log().mean())

            if bool(l.isnan()):
                raise ArithmeticError('encountered nan at iteration ' + str(int(n)) + ' batch ' + str(int(ns)))
            if track:
                print('c = ', str(float(l)), ' at iteration ', str(int(n)) + ' batch ' + str(int(ns)))

            optimizer[ns].zero_grad()
            l.backward()
            optimizer[ns].step()
            obj[ns].append(float(l))

    return {'model': model,
            'optimizer': optimizer,
            'cost': obj,
            'bound': boundaries}


def train_alg_mfc_fb_ot(data, lr=0.001, n_layers=2,
                        n_sample=100, n_iter=128, nt_subgrid=10,
                        s1=1, s2=1,
                        h=None, k=5, lock_dist=0.01,
                        r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1,
                        reg=1, reg1=1, reg2=2,
                        track=False, **_):

    t_data = data.time.unique()
    t_data.sort()
    nt_data = t_data.shape[0]
    n_ref = n_sample * 3
    dirac = np.repeat(1, n_ref) / n_ref
    tmap = []
    tmap_norm = []
    start_loc = []
    end_loc = []
    model_f = []
    model_b = []
    optimizer_f = []
    optimizer_b = []
    obj_f = []
    obj_b = []
    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2),
                                                                    torch.tensor(np.diag([s1, s2])).float())
    for i in range(nt_data - 1):
        data0 = data[data.time == t_data[i]][['x', 'y']].sample(n_ref, replace=True).to_numpy()
        data1 = data[data.time == t_data[i + 1]][['x', 'y']].sample(n_ref, replace=True).to_numpy()
        costm = ot_num.compute_dist(data0, data1, dim=2, single=False)
        start_loc.append(data0)
        end_loc.append(data1)
        reg_list = (100 - reg) * np.exp(-np.arange(100)) + reg
        # tmap_temp = ot_num.ot_unbalanced_log_stabilized(dirac, dirac, costm, reg, reg1, reg2, reg_list=reg_list)
        tmap_temp = ot_num.ot_balanced_log_stabilized(dirac, dirac, costm, reg, reg_list=reg_list)
        tmap.append(tmap_temp)
        tmap_norm_temp = tmap_temp.copy()
        tmap_norm_temp = np.diag(1 / tmap_norm_temp.sum(axis=1)) @ tmap_norm_temp
        tmap_norm.append(tmap_norm_temp)
        # end_loc.append(tmap_norm_temp @ data1)
        # end_loc.append(data1[tmap_norm_temp.argmax(axis=1)])
        model_f.append(NeuralNetwork(3, 2, 100, n_layers=n_layers))
        model_b.append(NeuralNetwork(3, 2, 100, n_layers=n_layers))
        optimizer_f.append(torch.optim.Adam(model_f[i].parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5))
        optimizer_b.append(torch.optim.Adam(model_b[i].parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5))
        obj_f.append([])
        obj_b.append([])
        t0 = t_data[i]
        t1 = t_data[i + 1]
        t_grid = np.linspace(t0, t1, nt_subgrid + 1)
        t_forward = t_grid[:-1]
        t_backward = t_grid[-1:0:-1]
        dt = 1 / nt_subgrid
        z0 = torch.tensor(data0)
        zf = torch.tensor(data1)
        for n in range(n_iter):
            l = torch.tensor(0.)
            x = torch.tensor(data[data.time == t_data[i]][['x', 'y']].sample(n_sample, replace=True).to_numpy())
            ### forward model
            for t in t_forward:
                inp = torch.cat([x, (t - t0) * torch.ones(n_sample, 1) / (t1 - t0)], dim=1)
                v = model_f[i](inp)
                e = me.sample([n_sample])
                x = x + v * dt + np.sqrt(dt) * e
                phat = kernel(x, h=h)
                l = l + r_v * v.pow(2).sum(axis=1).mean() * dt
                l = l + r_ent * phat.log().mean() * dt
            l = l - r_ent * phat.log().mean() * dt
            # p = kernel_pred(zf, x, h=h)
            # l = l + r_kl * (phat.log() - p.log()).mean()
            c1 = x[:, 0].reshape(-1, 1) - zf[:, 0]
            c2 = x[:, 1].reshape(-1, 1) - zf[:, 1]
            c = c1.pow(2) + c2.pow(2)
            c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
            # ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
            l = l + r_lock * (torch.max(c_lowk - lock_dist, torch.tensor(0.))).sum(axis=1).mean()
            # l = l + r_lock * (torch.max(ctr_lowk - lock_dist, torch.tensor(0.))).sum(axis=0).mean()
            if np.isnan(float(l)) or np.isinf(float(l)):
                raise ArithmeticError('encountered nan/inf at iteration ', str(int(n)) + ' forward time index ' + str(int(i)))
            if track:
                print('c = ', str(float(l)), ' at iteration ', str(int(n)) + ' forward time index ' + str(int(i)))
            optimizer_f[i].zero_grad()
            l.backward()
            optimizer_f[i].step()
            obj_f[i].append(float(l))
            ly = torch.tensor(0.)
            y = torch.tensor(data[data.time == t_data[i + 1]][['x', 'y']].sample(n_sample, replace=True).to_numpy())
            ### backward model
            for t in t_backward:
                inp_y = torch.cat([y, (t - t0) * torch.ones(n_sample, 1) / (t1 - t0)], dim=1)
                v_y = model_b[i](inp_y)
                e = me.sample([n_sample])
                y = y + v_y * dt + np.sqrt(dt) * e
                phat_y = kernel(y, h=h)
                ly = ly + r_v * v_y.pow(2).sum(axis=1).mean() * dt
                ly = ly + r_ent * phat_y.log().mean() * dt
            ly = ly - r_ent * phat_y.log().mean() * dt
            # p_y = kernel_pred(z0, y, h=h)
            # ly = ly + r_kl * (phat_y.log() - p_y.log()).mean()
            c1_y = y[:, 0].reshape(-1, 1) - z0[:, 0]
            c2_y = y[:, 1].reshape(-1, 1) - z0[:, 1]
            c_y = c1_y.pow(2) + c2_y.pow(2)
            c_y_lowk, c_y_rank = c_y.topk(k=k, dim=1, largest=False)
            # ctr_y_lowk, ctr_y_rank = c_y.topk(k=k, dim=0, largest=False)
            l = l + r_lock * (torch.max(c_y_lowk - lock_dist, torch.tensor(0.))).sum(axis=1).mean()
            # l = l + r_lock * (torch.max(ctr_y_lowk - lock_dist, torch.tensor(0.))).sum(axis=0).mean()
            if np.isnan(float(ly)) or np.isinf(float(ly)):
                raise ArithmeticError('encountered nan/inf at iteration ', str(int(n)) + ' backward time index ' + str(int(i)))
            if track:
                print('c = ', str(float(ly)), ' at iteration ', str(int(n)) + ' backward time index ' + str(int(i)))
            optimizer_b[i].zero_grad()
            ly.backward()
            optimizer_b[i].step()
            obj_b[i].append(float(ly))

    return {'model_f': model_f,
            'model_b': model_b,
            'opt_f': optimizer_f,
            'opt_b': optimizer_b,
            'obj_f': obj_f,
            'obj_b': obj_b,
            'tmap': tmap,
            'tmap_norm': tmap_norm,
            't_data': t_data,
            'x0': start_loc,
            'x1': end_loc}


def train_alg_mfc_fb_ot_gpu(data, lr=0.001, n_layers=2,
                            n_sample=100, n_iter=128, nt_subgrid=10,
                            s1=1, s2=1,
                            h=None, k=5, lock_dist=0.01,
                            r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1,
                            reg=1, reg1=1, reg2=2,
                            track=False, **_):

    t_data = data.time.unique()
    t_data.sort()
    nt_data = t_data.shape[0]
    n_ref = n_sample * 3
    dirac = np.repeat(1, n_ref) / n_ref
    tmap = []
    tmap_norm = []
    start_loc = []
    end_loc = []
    model_f = []
    model_b = []
    optimizer_f = []
    optimizer_b = []
    obj_f = []
    obj_b = []
    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2).cuda(),
                                                                    torch.tensor(np.diag([s1, s2])).cuda().float())
    for i in range(nt_data - 1):
        data0 = data[data.time == t_data[i]][['x', 'y']].sample(n_ref, replace=True).to_numpy()
        data1 = data[data.time == t_data[i + 1]][['x', 'y']].sample(n_ref, replace=True).to_numpy()
        costm = ot_num.compute_dist(data0, data1, dim=2, single=False)
        start_loc.append(data0)
        end_loc.append(data1)
        reg_list = (100 - reg) * np.exp(-np.arange(100)) + reg
        # tmap_temp = ot_num.ot_unbalanced_log_stabilized(dirac, dirac, costm, reg, reg1, reg2, reg_list=reg_list)
        tmap_temp = ot_num.ot_balanced_log_stabilized(dirac, dirac, costm, reg, reg_list=reg_list)
        tmap.append(tmap_temp)
        tmap_norm_temp = tmap_temp.copy()
        tmap_norm_temp = np.diag(1 / tmap_norm_temp.sum(axis=1)) @ tmap_norm_temp
        tmap_norm.append(tmap_norm_temp)
        # end_loc.append(tmap_norm_temp @ data1)
        # end_loc.append(data1[tmap_norm_temp.argmax(axis=1)])
        model_f.append(NeuralNetwork(3, 2, 100, n_layers=n_layers).cuda())
        model_b.append(NeuralNetwork(3, 2, 100, n_layers=n_layers).cuda())
        optimizer_f.append(torch.optim.Adam(model_f[i].parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5))
        optimizer_b.append(torch.optim.Adam(model_b[i].parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5))
        obj_f.append([])
        obj_b.append([])
        t0 = t_data[i]
        t1 = t_data[i + 1]
        t_grid = np.linspace(t0, t1, nt_subgrid + 1)
        t_forward = t_grid[:-1]
        t_backward = t_grid[-1:0:-1]
        dt = 1 / nt_subgrid
        z0 = torch.tensor(data0).cuda()
        zf = torch.tensor(data1).cuda()
        for n in range(n_iter):
            l = torch.tensor(0.).cuda()
            x = torch.tensor(data[data.time == t_data[i]][['x', 'y']].sample(n_sample, replace=True).to_numpy()).cuda()
            ### forward model
            for t in t_forward:
                inp = torch.cat([x, (t - t0) * torch.ones(n_sample, 1).cuda() / (t1 - t0)], dim=1)
                v = model_f[i](inp)
                e = me.sample([n_sample])
                x = x + v * dt + np.sqrt(dt) * e
                phat = kernel_gpu(x, h=h)
                l = l + r_v * v.pow(2).sum(axis=1).mean() * dt
                l = l + r_ent * phat.log().mean() * dt
            l = l - r_ent * phat.log().mean() * dt
            # p = kernel_pred(zf, x, h=h)
            # l = l + r_kl * (phat.log() - p.log()).mean()
            c1 = x[:, 0].reshape(-1, 1) - zf[:, 0]
            c2 = x[:, 1].reshape(-1, 1) - zf[:, 1]
            c = c1.pow(2) + c2.pow(2)
            c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
            # ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
            l = l + r_lock * (torch.max(c_lowk - lock_dist, torch.tensor(0.).cuda())).sum(axis=1).mean()
            # l = l + r_lock * (torch.max(ctr_lowk - lock_dist, torch.tensor(0.))).sum(axis=0).mean()
            if np.isnan(float(l)) or np.isinf(float(l)):
                raise ArithmeticError('encountered nan/inf at iteration ', str(int(n)) + ' forward time index ' + str(int(i)))
            if track:
                print('c = ', str(float(l)), ' at iteration ', str(int(n)) + ' forward time index ' + str(int(i)))
            optimizer_f[i].zero_grad()
            l.backward()
            optimizer_f[i].step()
            obj_f[i].append(float(l))
            ly = torch.tensor(0.).cuda()
            y = torch.tensor(data[data.time == t_data[i + 1]][['x', 'y']].sample(n_sample, replace=True).to_numpy()).cuda()
            ### backward model
            for t in t_backward:
                inp_y = torch.cat([y, (t - t0) * torch.ones(n_sample, 1).cuda() / (t1 - t0)], dim=1)
                v_y = model_b[i](inp_y)
                e = me.sample([n_sample])
                y = y + v_y * dt + np.sqrt(dt) * e
                phat_y = kernel_gpu(y, h=h)
                ly = ly + r_v * v_y.pow(2).sum(axis=1).mean() * dt
                ly = ly + r_ent * phat_y.log().mean() * dt
            ly = ly - r_ent * phat_y.log().mean() * dt
            # p_y = kernel_pred(z0, y, h=h)
            # ly = ly + r_kl * (phat_y.log() - p_y.log()).mean()
            c1_y = y[:, 0].reshape(-1, 1) - z0[:, 0]
            c2_y = y[:, 1].reshape(-1, 1) - z0[:, 1]
            c_y = c1_y.pow(2) + c2_y.pow(2)
            c_y_lowk, c_y_rank = c_y.topk(k=k, dim=1, largest=False)
            # ctr_y_lowk, ctr_y_rank = c_y.topk(k=k, dim=0, largest=False)
            l = l + r_lock * (torch.max(c_y_lowk - lock_dist, torch.tensor(0.).cuda())).sum(axis=1).mean()
            # l = l + r_lock * (torch.max(ctr_y_lowk - lock_dist, torch.tensor(0.).cuda())).sum(axis=0).mean()
            if np.isnan(float(ly)) or np.isinf(float(ly)):
                raise ArithmeticError('encountered nan/inf at iteration ', str(int(n)) + ' backward time index ' + str(int(i)))
            if track:
                print('c = ', str(float(ly)), ' at iteration ', str(int(n)) + ' backward time index ' + str(int(i)))
            optimizer_b[i].zero_grad()
            ly.backward()
            optimizer_b[i].step()
            obj_b[i].append(float(ly))

    return {'model_f': model_f,
            'model_b': model_b,
            'opt_f': optimizer_f,
            'opt_b': optimizer_b,
            'obj_f': obj_f,
            'obj_b': obj_b,
            'tmap': tmap,
            'tmap_norm': tmap_norm,
            't_data': t_data,
            'x0': start_loc,
            'x1': end_loc}


def train_alg_mfc_mixed(data, T, lr=0.001, n_mixed=5, n_layers=2,
                        n_sample=100, n_iter=128, nt_grid=100,
                        s1=1, s2=1,
                        h=None, k=5, lock_dist=0.01,
                        r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1,
                        track=False, **_):

    t_data = data.time.unique()
    t_data.sort()
    t_grid = np.unique(np.concatenate((t_data, np.linspace(0, T, nt_grid)), axis=None))
    t_grid.sort()
    nt = t_grid.shape[0]
    M = n_mixed

    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2),
                                                                    torch.tensor(np.diag([s1, s2])).float())

    model_list = []
    optimizer_list = []
    for m in range(M):
        model_list.append(NeuralNetwork(3, 2, 128, n_layers=n_layers))
        optimizer_list.append(torch.optim.Adam(model_list[m].parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5))

    model_mn = NeuralNetwork(3, M, 128, n_layers=n_layers)
    optimizer_mn = torch.optim.Adam(model_mn.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)
    weight_trans = nn.Softmin(dim=1)

    x0 = data[data.time == 0]

    # x_tensor = []
    # for t in t_data:
    #     x_tensor.append(torch.from_numpy(data[data.time == t][['x','y']].to_numpy()))

    obj = []

    for n in range(n_iter):

        # iteration for drift term
        x = torch.from_numpy(x0.sample(n_sample, replace=True)[['x','y']].to_numpy())
        ind_check = 1
        check = True
        l = torch.tensor(0.)
        for t_ind in range(nt - 1):
            ti = t_grid[t_ind]
            tf = t_grid[t_ind + 1]
            dt = (tf - ti) / T
            inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
            traj_w = weight_trans(model_mn(inp))
            v = torch.zeros(n_sample, 2)
            traj_id = torch.zeros(n_sample, M)
            for n_ind in range(n_sample):
                traj_pick = np.random.choice(np.arange(M), p=traj_w[n_ind, :].detach().numpy())
                traj_id[n_ind, traj_pick] = 1
            for m in range(M):
                # v = v + torch.diag(traj_w[:, m]) @ model_list[m](inp)
                v = v + torch.diag(traj_id[:, m]) @ model_list[m](inp)
            e = me.sample([n_sample])
            x = x + v * dt + np.sqrt(dt) * e
            l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
            phat = kernel(x, h=h)
            pvhat = kernel(v, h=h)
            l = l + r_ent_v * dt * (pvhat.log().mean())
            if check:
                if tf == t_data[ind_check]:
                    # x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample).to_numpy())
                    x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(10 * n_sample, replace=True).to_numpy())
                    c1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
                    c2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
                    c = c1.pow(2) + c2.pow(2)
                    weight_pen = weight_trans(model_mn(torch.cat([x_check, ti * torch.ones(x_check.shape[0], 1) / T], dim=1)))
                    c_pen = traj_w * traj_w.log() @ torch.ones(M, x_check.shape[0]) - traj_w @ weight_pen.log().T
                    c_pen = c_pen + (weight_pen * weight_pen.log() @ torch.ones(M, n_sample) - weight_pen @ traj_w.log().T).T
                    c_pen = c * c_pen.exp()
                    cpen_lowk, cpen_rank = c_pen.topk(k=k,dim=1, largest=False)
                    ctrpen_lowk, ctrpen_rank = c_pen.topk(k=k, dim=0, largest=False)
                    c_lowk = torch.gather(c, dim=1, index=cpen_rank)
                    ctr_lowk = torch.gather(c, dim=0, index=ctrpen_rank)
                    p = kernel_pred(x_check, x, h=h)
                    l = l - r_kl * p.log().mean()
                    c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                    ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                    l = l + r_lock * (torch.max(c_lowk - lock_dist, torch.tensor(0.))).sum(axis=1).mean()
                    l = l + r_lock * (torch.max(ctr_lowk - lock_dist, torch.tensor(0.))).sum(axis=0).mean()
                    if tf == t_data[-1]:
                        check = False
                    ind_check += 1
                else:
                    l = l + r_ent * dt * (phat.log().mean())
                # l = l + r_ent * dt * (phat.log().mean())

        # if bool(l.isnan()):
        #     raise ArithmeticError('encountered nan at iteration ' + str(int(n)) + ' for drift term')
        # if track:
        #     print('c = ', str(float(l)), ' at iteration ', str(int(n)) + ' for drift term')
        # for m in range(M):
        #     optimizer_list[m].zero_grad()
        # l.backward()
        # for m in range(M):
        #     optimizer_list[m].step()
        # obj.append(float(l))

        if bool(l.isnan()):
            raise ArithmeticError('encountered nan at iteration ' + str(int(n)))
        if track:
            print('c = ', str(float(l)), ' at iteration ', str(int(n)))
        for m in range(M):
            optimizer_list[m].zero_grad()
        optimizer_mn.zero_grad()
        l.backward()
        for m in range(M):
            optimizer_list[m].step()
        optimizer_mn.step()
        obj.append(float(l))

        # # iteration for weight term
        # x = torch.from_numpy(x0.sample(n_sample, replace=True)[['x','y']].to_numpy())
        # ind_check = 1
        # check = True
        # l = torch.tensor(0.)
        # for t_ind in range(nt - 1):
        #     ti = t_grid[t_ind]
        #     tf = t_grid[t_ind + 1]
        #     dt = (tf - ti) / T
        #     inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
        #     traj_w = weight_trans(model_mn(inp))
        #     v = torch.zeros(n_sample, 2)
        #     for m in range(M):
        #         v = v + torch.diag(traj_w[:, m]) @ model_list[m](inp)
        #     e = me.sample([n_sample])
        #     x = x + v * dt + np.sqrt(dt) * e
        #     l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
        #     phat = kernel(x, h=h)
        #     pvhat = kernel(v, h=h)
        #     l = l + r_ent_v * dt * (pvhat.log().mean())
        #     if check:
        #         if tf == t_data[ind_check]:
        #             x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample).to_numpy())
        #             # x_support = torch.from_numpy(data.sample(1000)[['x', 'y']].to_numpy())
        #             c1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
        #             c2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
        #             c = c1.pow(2) + c2.pow(2)
        #             weight_pen = weight_trans(model_mn(torch.cat([x_check, ti * torch.ones(n_sample, 1) / T], dim=1)))
        #             c_pen = traj_w @ traj_w.log().T - traj_w @ weight_pen.log().T
        #             c = c * torch.exp(c_pen)
        #             p = kernel_pred(x_check, x, h=h)
        #             l = l + r_kl * (phat.log() - p.log()).mean()
        #             c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
        #             # ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
        #             l = l + r_lock * (torch.max(c_lowk - lock_dist, torch.tensor(0.))).sum(axis=1).mean()
        #             # l = l + r_lock * (torch.max(ctr_lowk - lock_dist, torch.tensor(0.))).sum(axis=0).mean()
        #             if tf == t_data[-1]:
        #                 check = False
        #             ind_check += 1
        #         else:
        #             l = l + r_ent * dt * (phat.log().mean())
        #         # l = l + r_ent * dt * (phat.log().mean())

        # if bool(l.isnan()):
        #     raise ArithmeticError('encountered nan at iteration ' + str(int(n)) + ' for weight term')
        # if track:
        #     print('c = ', str(float(l)), ' at iteration ', str(int(n)) + ' for weight term')

        # optimizer_mn.zero_grad()
        # l.backward()
        # optimizer_mn.step()
        # obj.append(float(l))


    return {'model_drift': model_list,
            'optimizer_drift': optimizer_list,
            'model_mn': model_mn,
            'optimizer_mn': optimizer_mn,
            'cost': obj}


def train_alg_mfc_mixed_gpu(data, T, lr=0.001, n_mixed=5, n_layers=2,
                            n_sample=100, n_iter=128, nt_grid=100,
                            s1=1, s2=1,
                            h=None, k=5, lock_dist=0.01,
                            r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1,
                            track=False, **_):

    t_data = data.time.unique()
    t_data.sort()
    t_grid = np.unique(np.concatenate((t_data, np.linspace(0, T, nt_grid)), axis=None))
    t_grid.sort()
    nt = t_grid.shape[0]
    M = n_mixed

    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2).cuda(),
                                                                    torch.tensor(np.diag([s1, s2])).cuda().float())

    model_list = []
    optimizer_list = []
    for m in range(M):
        model_list.append(NeuralNetwork(3, 2, 128, n_layers=n_layers).cuda())
        optimizer_list.append(torch.optim.Adam(model_list[m].parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5))

    model_mn = NeuralNetwork(3, M, 128, n_layers=n_layers).cuda()
    optimizer_mn = torch.optim.Adam(model_mn.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)
    weight_trans = nn.Softmin(dim=1)

    x0 = data[data.time == 0]

    # x_tensor = []
    # for t in t_data:
    #     x_tensor.append(torch.from_numpy(data[data.time == t][['x','y']].to_numpy()))

    obj = []

    for n in range(n_iter):

        # iteration for drift term
        x = torch.from_numpy(x0.sample(n_sample, replace=True)[['x','y']].to_numpy()).cuda()
        ind_check = 1
        check = True
        l = torch.tensor(0.).cuda()
        for t_ind in range(nt - 1):
            ti = t_grid[t_ind]
            tf = t_grid[t_ind + 1]
            dt = (tf - ti) / T
            inp = torch.cat([x, ti * torch.ones(n_sample, 1).cuda() / T], dim=1)
            traj_w = weight_trans(model_mn(inp))
            v = torch.zeros(n_sample, 2).cuda()
            traj_id = torch.zeros(n_sample, M).cuda()
            for n_ind in range(n_sample):
                traj_pick = np.random.choice(np.arange(M), p=traj_w[n_ind, :].cpu().detach().numpy())
                traj_id[n_ind, traj_pick] = 1
            for m in range(M):
                # v = v + torch.diag(traj_w[:, m]) @ model_list[m](inp)
                v = v + torch.diag(traj_id[:, m]) @ model_list[m](inp)
            e = me.sample([n_sample])
            x = x + v * dt + np.sqrt(dt) * e
            l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
            phat = kernel_gpu(x, h=h)
            pvhat = kernel_gpu(v, h=h)
            l = l + r_ent_v * dt * (pvhat.log().mean())
            if check:
                if tf == t_data[ind_check]:
                    # x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample).to_numpy()).cuda()
                    x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(10 * n_sample, replace=True).to_numpy()).cuda()
                    c1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
                    c2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
                    c = c1.pow(2) + c2.pow(2)
                    weight_pen = weight_trans(model_mn(torch.cat([x_check, ti * torch.ones(x_check.shape[0], 1).cuda() / T], dim=1)))
                    c_pen = traj_w * traj_w.log() @ torch.ones(M, x_check.shape[0]).cuda() - traj_w @ weight_pen.log().T
                    c_pen = c_pen + (weight_pen * weight_pen.log() @ torch.ones(M, n_sample).cuda() - weight_pen @ traj_w.log().T).T
                    c_pen = c * c_pen.exp()
                    cpen_lowk, cpen_rank = c_pen.topk(k=k,dim=1, largest=False)
                    ctrpen_lowk, ctrpen_rank = c_pen.topk(k=k, dim=0, largest=False)
                    c_lowk = torch.gather(c, dim=1, index=cpen_rank)
                    ctr_lowk = torch.gather(c, dim=0, index=ctrpen_rank)
                    p = kernel_pred_gpu(x_check, x, h=h)
                    l = l - r_kl * p.log().mean()
                    c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                    ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                    l = l + r_lock * (torch.max(c_lowk - lock_dist, torch.tensor(0.).cuda())).sum(axis=1).mean()
                    l = l + r_lock * (torch.max(ctr_lowk - lock_dist, torch.tensor(0.).cuda())).sum(axis=0).mean()
                    if tf == t_data[-1]:
                        check = False
                    ind_check += 1
                else:
                    l = l + r_ent * dt * (phat.log().mean())
                # l = l + r_ent * dt * (phat.log().mean())

        # if bool(l.isnan()):
        #     raise ArithmeticError('encountered nan at iteration ' + str(int(n)) + ' for drift term')
        # if track:
        #     print('c = ', str(float(l)), ' at iteration ', str(int(n)) + ' for drift term')
        # for m in range(M):
        #     optimizer_list[m].zero_grad()
        # l.backward()
        # for m in range(M):
        #     optimizer_list[m].step()
        # obj.append(float(l))

        if bool(l.isnan()):
            raise ArithmeticError('encountered nan at iteration ' + str(int(n)))
        if track:
            print('c = ', str(float(l)), ' at iteration ', str(int(n)))
        for m in range(M):
            optimizer_list[m].zero_grad()
        optimizer_mn.zero_grad()
        l.backward()
        for m in range(M):
            optimizer_list[m].step()
        optimizer_mn.step()
        obj.append(float(l))

        # # iteration for weight term
        # x = torch.from_numpy(x0.sample(n_sample, replace=True)[['x','y']].to_numpy())
        # ind_check = 1
        # check = True
        # l = torch.tensor(0.)
        # for t_ind in range(nt - 1):
        #     ti = t_grid[t_ind]
        #     tf = t_grid[t_ind + 1]
        #     dt = (tf - ti) / T
        #     inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
        #     traj_w = weight_trans(model_mn(inp))
        #     v = torch.zeros(n_sample, 2)
        #     for m in range(M):
        #         v = v + torch.diag(traj_w[:, m]) @ model_list[m](inp)
        #     e = me.sample([n_sample])
        #     x = x + v * dt + np.sqrt(dt) * e
        #     l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
        #     phat = kernel(x, h=h)
        #     pvhat = kernel(v, h=h)
        #     l = l + r_ent_v * dt * (pvhat.log().mean())
        #     if check:
        #         if tf == t_data[ind_check]:
        #             x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample).to_numpy())
        #             # x_support = torch.from_numpy(data.sample(1000)[['x', 'y']].to_numpy())
        #             c1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
        #             c2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
        #             c = c1.pow(2) + c2.pow(2)
        #             weight_pen = weight_trans(model_mn(torch.cat([x_check, ti * torch.ones(n_sample, 1) / T], dim=1)))
        #             c_pen = traj_w @ traj_w.log().T - traj_w @ weight_pen.log().T
        #             c = c * torch.exp(c_pen)
        #             p = kernel_pred(x_check, x, h=h)
        #             l = l + r_kl * (phat.log() - p.log()).mean()
        #             c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
        #             # ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
        #             l = l + r_lock * (torch.max(c_lowk - lock_dist, torch.tensor(0.))).sum(axis=1).mean()
        #             # l = l + r_lock * (torch.max(ctr_lowk - lock_dist, torch.tensor(0.))).sum(axis=0).mean()
        #             if tf == t_data[-1]:
        #                 check = False
        #             ind_check += 1
        #         else:
        #             l = l + r_ent * dt * (phat.log().mean())
        #         # l = l + r_ent * dt * (phat.log().mean())

        # if bool(l.isnan()):
        #     raise ArithmeticError('encountered nan at iteration ' + str(int(n)) + ' for weight term')
        # if track:
        #     print('c = ', str(float(l)), ' at iteration ', str(int(n)) + ' for weight term')

        # optimizer_mn.zero_grad()
        # l.backward()
        # optimizer_mn.step()
        # obj.append(float(l))


    return {'model_drift': model_list,
            'optimizer_drift': optimizer_list,
            'model_mn': model_mn,
            'optimizer_mn': optimizer_mn,
            'cost': obj}


def train_alg_mfc_fb_mixed(data, T, lr=0.001, n_mixed=5, n_layers=2,
                           n_sample=100, n_iter=128, nt_grid=100, fb_iter=5,
                           s1=1, s2=1,
                           h=None, k=5, lock_dist=0.01, use_score=False,
                           r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1,
                           track=False, **_):

    t_data = data.time.unique()
    t_data.sort()
    t_grid = np.unique(np.concatenate((t_data, np.linspace(0, T, nt_grid)), axis=None))
    t_grid.sort()
    nt = t_grid.shape[0]
    M = n_mixed

    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2),
                                                                    torch.tensor(np.diag([s1, s2])).float())

    weight_trans = nn.Softmin(dim=1)

    model_f_list = []
    optimizer_f_list = []
    for m in range(M):
        model_f_list.append(NeuralNetwork(3, 2, 128, n_layers=n_layers))
        optimizer_f_list.append(torch.optim.Adam(model_f_list[m].parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5))

    model_f_mn = NeuralNetwork(3, M, 128, n_layers=n_layers)
    optimizer_f_mn = torch.optim.Adam(model_f_mn.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

    model_b_list = []
    optimizer_b_list = []
    for m in range(M):
        model_b_list.append(NeuralNetwork(3, 2, 128, n_layers=n_layers))
        optimizer_b_list.append(torch.optim.Adam(model_b_list[m].parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5))

    model_b_mn = NeuralNetwork(3, M, 128, n_layers=n_layers)
    optimizer_b_mn = torch.optim.Adam(model_b_mn.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

    x0 = data[data.time == 0]
    x1 = data[data.time == data.time.max()]

    obj_f = []
    obj_b = []

    x_target_track = []
    x_start = x0.sample(n_sample, replace=True)[['x','y']].to_numpy()

    for i_fb in range(fb_iter):

        obj = []

        if i_fb > 0:
            x_target = x_target_track[-1]

        if i_fb % 2 == 0: # forward modeling

            for n in range(n_iter):

                # iteration for drift term
                x = torch.from_numpy(x_start)
                ind_check = 1
                check = True
                l = torch.tensor(0.)
                for t_ind in range(nt - 1):
                    ti = t_grid[t_ind]
                    tf = t_grid[t_ind + 1]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
                    traj_w = weight_trans(model_f_mn(inp))
                    v = torch.zeros(n_sample, 2)
                    traj_id = torch.zeros(n_sample, M)
                    for n_ind in range(n_sample):
                        traj_pick = np.random.choice(np.arange(M), p=traj_w[n_ind, :].detach().numpy())
                        traj_id[n_ind, traj_pick] = 1
                    for m in range(M):
                        # v = v + torch.diag(traj_w[:, m]) @ model_f_list[m](inp)
                        v = v + torch.diag(traj_id[:, m]) @ model_f_list[m](inp)
                    e = me.sample([n_sample])
                    x = x + v * dt + np.sqrt(dt) * e
                    l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
                    phat = kernel(x, h=h)
                    pvhat = kernel(v, h=h)
                    l = l + r_ent_v * dt * (pvhat.log().mean())
                    if check:
                        if tf == t_data[ind_check]:
                            # x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample).to_numpy())
                            x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(10 * n_sample, replace=True).to_numpy())
                            c1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
                            c2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
                            c = c1.pow(2) + c2.pow(2)
                            weight_pen = weight_trans(model_f_mn(torch.cat([x_check, ti * torch.ones(x_check.shape[0], 1) / T], dim=1)))
                            c_pen = traj_w * traj_w.log() @ torch.ones(M, x_check.shape[0]) - traj_w @ weight_pen.log().T
                            c_pen = c_pen + (weight_pen * weight_pen.log() @ torch.ones(M, n_sample) - weight_pen @ traj_w.log().T).T
                            c_pen = c * c_pen.exp()
                            cpen_lowk, cpen_rank = c_pen.topk(k=k,dim=1, largest=False)
                            ctrpen_lowk, ctrpen_rank = c_pen.topk(k=k, dim=0, largest=False)
                            c_lowk = torch.gather(c, dim=1, index=cpen_rank)
                            ctr_lowk = torch.gather(c, dim=0, index=ctrpen_rank)
                            p = kernel_pred(x_check, x, h=h)
                            l = l - r_kl * p.log().mean()
                            c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                            ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                            l = l + r_lock * (torch.max(c_lowk - lock_dist, torch.tensor(0.))).sum(axis=1).mean()
                            l = l + r_lock * (torch.max(ctr_lowk - lock_dist, torch.tensor(0.))).sum(axis=0).mean()
                            if tf == t_data[-1]:
                                check = False
                            ind_check += 1
                        else:
                            l = l + r_ent * dt * (phat.log().mean())
                    if i_fb > 0:
                        x_ref = torch.tensor(x_target[t_ind + 1])
                        l = l + (x - x_ref).pow(2).sum(axis=1).mean()
                if bool(l.isnan()):
                    raise ArithmeticError('encountered nan at forward iteration ' + str(i_fb) + ' iteration ' + str(int(n)))
                if track:
                    print('c = ', str(float(l)), ' at forward iteration ' + str(i_fb) + ' iteration ' + str(int(n)))

                for m in range(M):
                    optimizer_f_list[m].zero_grad()
                optimizer_f_mn.zero_grad()
                l.backward()
                for m in range(M):
                    optimizer_f_list[m].step()
                optimizer_f_mn.step()
                obj.append(float(l))

            if use_score:
                x = torch.tensor(x1.sample(n_sample, replace=True)[['x','y']].to_numpy())
                x_start = x.detach().numpy()
                x_track_temp = []
                x_track_temp.append(x.detach().numpy())
                for t_ind in range(nt - 1):
                    ti = t_grid[-(t_ind + 1)]
                    tf = t_grid[-(t_ind + 2)]
                    dt = (ti - tf) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
                    traj_w = weight_trans(model_f_mn(inp))
                    v = torch.zeros(n_sample, 2)
                    traj_id = torch.zeros(n_sample, M)
                    for n_ind in range(n_sample):
                        traj_pick = np.random.choice(np.arange(M), p=traj_w[n_ind, :].detach().numpy())
                        traj_id[n_ind, traj_pick] = 1
                    for m in range(M):
                        # v = v + torch.diag(traj_w[:, m]) @ model_f_list[m](inp)
                        v = v + torch.diag(traj_id[:, m]) @ model_f_list[m](inp)
                    e = me.sample([n_sample])
                    score = score_est(x, x, h=h)
                    x = x + (v - s1 * score) * dt + np.sqrt(dt) * e
                    x_track_temp.append(x.detach().numpy())
                x_target_track.append(x_track_temp)
            else:
                x = torch.tensor(x0.sample(n_sample, replace=True)[['x']].to_numpy())
                x_track_temp = []
                x_track_temp.append(x.detach().numpy())
                for t_ind in range(nt - 1):
                    ti = t_grid[t_ind]
                    tf = t_grid[t_ind + 1]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
                    traj_w = weight_trans(model_f_mn(inp))
                    v = torch.zeros(n_sample, 2)
                    traj_id = torch.zeros(n_sample, M)
                    for n_ind in range(n_sample):
                        traj_pick = np.random.choice(np.arange(M), p=traj_w[n_ind, :].detach().numpy())
                        traj_id[n_ind, traj_pick] = 1
                    for m in range(M):
                        # v = v + torch.diag(traj_w[:, m]) @ model_f_list[m](inp)
                        v = v + torch.diag(traj_id[:, m]) @ model_f_list[m](inp)
                    e = me.sample([n_sample])
                    x = x + v * dt + np.sqrt(dt) * e
                    x_track_temp.append(x.detach().numpy())
                x_target_track.append(x_track_temp[::-1])
                x_start = x.detach().numpy()

            obj_f.append(obj)

        else: # backward modeling

            for n in range(n_iter):

                # iteration for drift term
                x = torch.from_numpy(x_start)
                ind_check = 1
                check = True
                l = torch.tensor(0.)
                for t_ind in range(nt - 1):
                    ti = T - t_grid[-(t_ind + 1)]
                    tf = T - t_grid[-(t_ind + 2)]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
                    traj_w = weight_trans(model_b_mn(inp))
                    v = torch.zeros(n_sample, 2)
                    traj_id = torch.zeros(n_sample, M)
                    for n_ind in range(n_sample):
                        traj_pick = np.random.choice(np.arange(M), p=traj_w[n_ind, :].detach().numpy())
                        traj_id[n_ind, traj_pick] = 1
                    for m in range(M):
                        # v = v + torch.diag(traj_w[:, m]) @ model_b_list[m](inp)
                        v = v + torch.diag(traj_id[:, m]) @ model_b_list[m](inp)
                    e = me.sample([n_sample])
                    x = x + v * dt + np.sqrt(dt) * e
                    l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
                    phat = kernel(x, h=h)
                    pvhat = kernel(-v, h=h)
                    l = l + r_ent_v * dt * (pvhat.log().mean())
                    if check:
                        if tf == t_data[-(ind_check + 1)]:
                            # x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample).to_numpy())
                            x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(10 * n_sample, replace=True).to_numpy())
                            c1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
                            c2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
                            c = c1.pow(2) + c2.pow(2)
                            weight_pen = weight_trans(model_b_mn(torch.cat([x_check, ti * torch.ones(x_check.shape[0], 1) / T], dim=1)))
                            c_pen = traj_w * traj_w.log() @ torch.ones(M, x_check.shape[0]) - traj_w @ weight_pen.log().T
                            c_pen = c_pen + (weight_pen * weight_pen.log() @ torch.ones(M, n_sample) - weight_pen @ traj_w.log().T).T
                            c_pen = c * c_pen.exp()
                            cpen_lowk, cpen_rank = c_pen.topk(k=k,dim=1, largest=False)
                            ctrpen_lowk, ctrpen_rank = c_pen.topk(k=k, dim=0, largest=False)
                            c_lowk = torch.gather(c, dim=1, index=cpen_rank)
                            ctr_lowk = torch.gather(c, dim=0, index=ctrpen_rank)
                            p = kernel_pred(x_check, x, h=h)
                            l = l - r_kl * p.log().mean()
                            c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                            ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                            l = l + r_lock * (torch.max(c_lowk - lock_dist, torch.tensor(0.))).sum(axis=1).mean()
                            l = l + r_lock * (torch.max(ctr_lowk - lock_dist, torch.tensor(0.))).sum(axis=0).mean()
                            if tf == t_data[0]:
                                check = False
                            ind_check += 1
                        else:
                            l = l + r_ent * dt * (phat.log().mean())
                    if i_fb > 0:
                        x_ref = torch.tensor(x_target[t_ind + 1])
                        l = l + (x - x_ref).pow(2).sum(axis=1).mean()
                if bool(l.isnan()):
                    raise ArithmeticError('encountered nan at backward iteration ' + str(i_fb) + ' iteration ' + str(int(n)))
                if track:
                    print('c = ', str(float(l)), ' at backward iteration ' + str(i_fb) + ' iteration ' + str(int(n)))
                for m in range(M):
                    optimizer_b_list[m].zero_grad()
                optimizer_b_mn.zero_grad()
                l.backward()
                for m in range(M):
                    optimizer_b_list[m].step()
                optimizer_b_mn.step()
                obj.append(float(l))

            if use_score:
                x = torch.tensor(x0.sample(n_sample, replace=True)[['x','y']].to_numpy())
                x_start = x.detach().numpy()
                x_track_temp = []
                x_track_temp.append(x.detach().numpy())
                for t_ind in range(nt - 1):
                    ti = t_grid[t_ind]
                    tf = t_grid[t_ind + 1]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, (T - ti) * torch.ones(n_sample, 1) / T], dim=1)
                    traj_w = weight_trans(model_b_mn(inp))
                    v = torch.zeros(n_sample, 2)
                    traj_id = torch.zeros(n_sample, M)
                    for n_ind in range(n_sample):
                        traj_pick = np.random.choice(np.arange(M), p=traj_w[n_ind, :].detach().numpy())
                        traj_id[n_ind, traj_pick] = 1
                    for m in range(M):
                        # v = v + torch.diag(traj_w[:, m]) @ model_b_list[m](inp)
                        v = v + torch.diag(traj_id[:, m]) @ model_b_list[m](inp)
                    e = me.sample([n_sample])
                    score = score_est(x, x, h=h)
                    x = x + (v + s1 * score) * dt + np.sqrt(dt) * e
                    x_track_temp.append(x.detach().numpy())
                x_target_track.append(x_track_temp)
            else:
                x = torch.tensor(x1.sample(n_sample, replace=True)[['x']].to_numpy())
                x_track_temp = []
                x_track_temp.append(x.detach().numpy())
                for t_ind in range(nt - 1):
                    ti = T - t_grid[-(t_ind + 1)]
                    tf = T - t_grid[-(t_ind + 2)]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
                    traj_w = weight_trans(model_b_mn(inp))
                    v = torch.zeros(n_sample, 2)
                    traj_id = torch.zeros(n_sample, M)
                    for n_ind in range(n_sample):
                        traj_pick = np.random.choice(np.arange(M), p=traj_w[n_ind, :].detach().numpy())
                        traj_id[n_ind, traj_pick] = 1
                    for m in range(M):
                        # v = v + torch.diag(traj_w[:, m]) @ model_b_list[m](inp)
                        v = v + torch.diag(traj_id[:, m]) @ model_b_list[m](inp)
                    e = me.sample([n_sample])
                    x = x + v * dt + np.sqrt(dt) * e
                    x_track_temp.append(x.detach().numpy())
                x_target_track.append(x_track_temp[::-1])
                x_start = x.detach().numpy()

            obj_b.append(obj)

    return {'model_f': model_f_list,
            'model_b': model_b_list,
            'optimizer_f': optimizer_f_list,
            'optimizer_b': optimizer_b_list,
            'model_f_mn': model_f_mn,
            'model_b_mn': model_b_mn,
            'optimizer_f_mn': optimizer_f_mn,
            'optimizer_b_mn': optimizer_b_mn,
            'cost_f': obj_f,
            'cost_b': obj_b}


def train_alg_mfc_fb_mixed_gpu(data, T, lr=0.001, n_mixed=5, n_layers=2,
                               n_sample=100, n_iter=128, nt_grid=100, fb_iter=5,
                               s1=1, s2=1,
                               h=None, k=5, lock_dist=0.01, use_score=False,
                               r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1,
                               track=False, **_):

    t_data = data.time.unique()
    t_data.sort()
    t_grid = np.unique(np.concatenate((t_data, np.linspace(0, T, nt_grid)), axis=None))
    t_grid.sort()
    nt = t_grid.shape[0]
    M = n_mixed

    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2).cuda(),
                                                                    torch.tensor(np.diag([s1, s2])).cuda().float())

    weight_trans = nn.Softmin(dim=1)

    model_f_list = []
    optimizer_f_list = []
    for m in range(M):
        model_f_list.append(NeuralNetwork(3, 2, 128, n_layers=n_layers).cuda())
        optimizer_f_list.append(torch.optim.Adam(model_f_list[m].parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5))

    model_f_mn = NeuralNetwork(3, M, 128, n_layers=n_layers).cuda()
    optimizer_f_mn = torch.optim.Adam(model_f_mn.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

    model_b_list = []
    optimizer_b_list = []
    for m in range(M):
        model_b_list.append(NeuralNetwork(3, 2, 128, n_layers=n_layers).cuda())
        optimizer_b_list.append(torch.optim.Adam(model_b_list[m].parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5))

    model_b_mn = NeuralNetwork(3, M, 128, n_layers=n_layers).cuda()
    optimizer_b_mn = torch.optim.Adam(model_b_mn.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

    x0 = data[data.time == 0]
    x1 = data[data.time == data.time.max()]

    obj_f = []
    obj_b = []

    x_target_track = []
    x_start = x0.sample(n_sample, replace=True)[['x','y']].to_numpy()

    for i_fb in range(fb_iter):

        obj = []

        if i_fb > 0:
            x_target = x_target_track[-1]

        if i_fb % 2 == 0: # forward modeling

            for n in range(n_iter):

                # iteration for drift term
                x = torch.from_numpy(x_start).cuda()
                ind_check = 1
                check = True
                l = torch.tensor(0.).cuda()
                for t_ind in range(nt - 1):
                    ti = t_grid[t_ind]
                    tf = t_grid[t_ind + 1]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1).cuda() / T], dim=1)
                    traj_w = weight_trans(model_f_mn(inp))
                    v = torch.zeros(n_sample, 2).cuda()
                    traj_id = torch.zeros(n_sample, M).cuda()
                    for n_ind in range(n_sample):
                        traj_pick = np.random.choice(np.arange(M), p=traj_w[n_ind, :].detach().numpy())
                        traj_id[n_ind, traj_pick] = 1
                    for m in range(M):
                        # v = v + torch.diag(traj_w[:, m]) @ model_f_list[m](inp)
                        v = v + torch.diag(traj_id[:, m]) @ model_f_list[m](inp)
                    e = me.sample([n_sample])
                    x = x + v * dt + np.sqrt(dt) * e
                    l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
                    phat = kernel_gpu(x, h=h)
                    pvhat = kernel_gpu(v, h=h)
                    l = l + r_ent_v * dt * (pvhat.log().mean())
                    if check:
                        if tf == t_data[ind_check]:
                            # x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample).to_numpy()).cuda()
                            x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(10 * n_sample, replace=True).to_numpy()).cuda()
                            c1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
                            c2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
                            c = c1.pow(2) + c2.pow(2)
                            weight_pen = weight_trans(model_f_mn(torch.cat([x_check, ti * torch.ones(x_check.shape[0], 1).cuda() / T], dim=1)))
                            c_pen = traj_w * traj_w.log() @ torch.ones(M, x_check.shape[0]).cuda() - traj_w @ weight_pen.log().T
                            c_pen = c_pen + (weight_pen * weight_pen.log() @ torch.ones(M, n_sample).cuda() - weight_pen @ traj_w.log().T).T
                            c_pen = c * c_pen.exp()
                            cpen_lowk, cpen_rank = c_pen.topk(k=k,dim=1, largest=False)
                            ctrpen_lowk, ctrpen_rank = c_pen.topk(k=k, dim=0, largest=False)
                            c_lowk = torch.gather(c, dim=1, index=cpen_rank)
                            ctr_lowk = torch.gather(c, dim=0, index=ctrpen_rank)
                            p = kernel_pred_gpu(x_check, x, h=h)
                            l = l - r_kl * p.log().mean()
                            c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                            ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                            l = l + r_lock * (torch.max(c_lowk - lock_dist, torch.tensor(0.).cuda())).sum(axis=1).mean()
                            l = l + r_lock * (torch.max(ctr_lowk - lock_dist, torch.tensor(0.).cuda())).sum(axis=0).mean()
                            if tf == t_data[-1]:
                                check = False
                            ind_check += 1
                        else:
                            l = l + r_ent * dt * (phat.log().mean())
                    if i_fb > 0:
                        x_ref = torch.tensor(x_target[t_ind + 1]).cuda()
                        l = l + (x - x_ref).pow(2).sum(axis=1).mean()
                if bool(l.isnan()):
                    raise ArithmeticError('encountered nan at forward iteration ' + str(i_fb) + ' iteration ' + str(int(n)))
                if track:
                    print('c = ', str(float(l)), ' at forward iteration ' + str(i_fb) + ' iteration ' + str(int(n)))

                for m in range(M):
                    optimizer_f_list[m].zero_grad()
                optimizer_f_mn.zero_grad()
                l.backward()
                for m in range(M):
                    optimizer_f_list[m].step()
                optimizer_f_mn.step()
                obj.append(float(l))

            if use_score:
                x = torch.tensor(x1.sample(n_sample, replace=True)[['x','y']].to_numpy()).cuda()
                x_start = x.cpu().detach().numpy()
                x_track_temp = []
                x_track_temp.append(x.cpu().detach().numpy())
                for t_ind in range(nt - 1):
                    ti = t_grid[-(t_ind + 1)]
                    tf = t_grid[-(t_ind + 2)]
                    dt = (ti - tf) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1).cuda() / T], dim=1)
                    traj_w = weight_trans(model_f_mn(inp))
                    v = torch.zeros(n_sample, 2).cuda()
                    traj_id = torch.zeros(n_sample, M).cuda()
                    for n_ind in range(n_sample):
                        traj_pick = np.random.choice(np.arange(M), p=traj_w[n_ind, :].detach().numpy())
                        traj_id[n_ind, traj_pick] = 1
                    for m in range(M):
                        # v = v + torch.diag(traj_w[:, m]) @ model_f_list[m](inp)
                        v = v + torch.diag(traj_id[:, m]) @ model_f_list[m](inp)
                    e = me.sample([n_sample])
                    score = score_est_gpu(x, x, h=h)
                    x = x + (v - s1 * score) * dt + np.sqrt(dt) * e
                    x_track_temp.append(x.cpu().detach().numpy())
                x_target_track.append(x_track_temp)
            else:
                x = torch.tensor(x0.sample(n_sample, replace=True)[['x']].to_numpy()).cuda()
                x_track_temp = []
                x_track_temp.append(x.cpu().detach().numpy())
                for t_ind in range(nt - 1):
                    ti = t_grid[t_ind]
                    tf = t_grid[t_ind + 1]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1).cuda() / T], dim=1)
                    traj_w = weight_trans(model_f_mn(inp))
                    v = torch.zeros(n_sample, 2).cuda()
                    traj_id = torch.zeros(n_sample, M).cuda()
                    for n_ind in range(n_sample):
                        traj_pick = np.random.choice(np.arange(M), p=traj_w[n_ind, :].detach().numpy())
                        traj_id[n_ind, traj_pick] = 1
                    for m in range(M):
                        # v = v + torch.diag(traj_w[:, m]) @ model_f_list[m](inp)
                        v = v + torch.diag(traj_id[:, m]) @ model_f_list[m](inp)
                    e = me.sample([n_sample])
                    x = x + v * dt + np.sqrt(dt) * e
                    x_track_temp.append(x.cpu().detach().numpy())
                x_target_track.append(x_track_temp[::-1])
                x_start = x.cpu().detach().numpy()

            obj_f.append(obj)

        else: # backward modeling

            for n in range(n_iter):

                # iteration for drift term
                x = torch.from_numpy(x_start).cuda()
                ind_check = 1
                check = True
                l = torch.tensor(0.)
                for t_ind in range(nt - 1):
                    ti = T - t_grid[-(t_ind + 1)]
                    tf = T - t_grid[-(t_ind + 2)]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1).cuda() / T], dim=1)
                    traj_w = weight_trans(model_b_mn(inp))
                    v = torch.zeros(n_sample, 2).cuda()
                    traj_id = torch.zeros(n_sample, M).cuda()
                    for n_ind in range(n_sample):
                        traj_pick = np.random.choice(np.arange(M), p=traj_w[n_ind, :].detach().numpy())
                        traj_id[n_ind, traj_pick] = 1
                    for m in range(M):
                        # v = v + torch.diag(traj_w[:, m]) @ model_b_list[m](inp)
                        v = v + torch.diag(traj_id[:, m]) @ model_b_list[m](inp)
                    e = me.sample([n_sample])
                    x = x + v * dt + np.sqrt(dt) * e
                    l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
                    phat = kernel_gpu(x, h=h)
                    pvhat = kernel_gpu(-v, h=h)
                    l = l + r_ent_v * dt * (pvhat.log().mean())
                    if check:
                        if tf == t_data[-(ind_check + 1)]:
                            # x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample).to_numpy()).cuda()
                            x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(10 * n_sample, replace=True).to_numpy()).cuda()
                            c1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
                            c2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
                            c = c1.pow(2) + c2.pow(2)
                            weight_pen = weight_trans(model_b_mn(torch.cat([x_check, ti * torch.ones(x_check.shape[0], 1).cuda() / T], dim=1)))
                            c_pen = traj_w * traj_w.log() @ torch.ones(M, x_check.shape[0]).cuda() - traj_w @ weight_pen.log().T
                            c_pen = c_pen + (weight_pen * weight_pen.log() @ torch.ones(M, n_sample).cuda() - weight_pen @ traj_w.log().T).T
                            c_pen = c * c_pen.exp()
                            cpen_lowk, cpen_rank = c_pen.topk(k=k,dim=1, largest=False)
                            ctrpen_lowk, ctrpen_rank = c_pen.topk(k=k, dim=0, largest=False)
                            c_lowk = torch.gather(c, dim=1, index=cpen_rank)
                            ctr_lowk = torch.gather(c, dim=0, index=ctrpen_rank)
                            p = kernel_pred_gpu(x_check, x, h=h)
                            l = l - r_kl * p.log().mean()
                            c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                            ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                            l = l + r_lock * (torch.max(c_lowk - lock_dist, torch.tensor(0.).cuda())).sum(axis=1).mean()
                            l = l + r_lock * (torch.max(ctr_lowk - lock_dist, torch.tensor(0.).cuda())).sum(axis=0).mean()
                            if tf == t_data[0]:
                                check = False
                            ind_check += 1
                        else:
                            l = l + r_ent * dt * (phat.log().mean())
                    if i_fb > 0:
                        x_ref = torch.tensor(x_target[t_ind + 1]).cuda()
                        l = l + (x - x_ref).pow(2).sum(axis=1).mean()
                if bool(l.isnan()):
                    raise ArithmeticError('encountered nan at backward iteration ' + str(i_fb) + ' iteration ' + str(int(n)))
                if track:
                    print('c = ', str(float(l)), ' at backward iteration ' + str(i_fb) + ' iteration ' + str(int(n)))
                for m in range(M):
                    optimizer_b_list[m].zero_grad()
                optimizer_b_mn.zero_grad()
                l.backward()
                for m in range(M):
                    optimizer_b_list[m].step()
                optimizer_b_mn.step()
                obj.append(float(l))

            if use_score:
                x = torch.tensor(x0.sample(n_sample, replace=True)[['x','y']].to_numpy()).cuda()
                x_start = x.cpu().detach().numpy()
                x_track_temp = []
                x_track_temp.append(x.cpu().detach().numpy())
                for t_ind in range(nt - 1):
                    ti = t_grid[t_ind]
                    tf = t_grid[t_ind + 1]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, (T - ti) * torch.ones(n_sample, 1).cuda() / T], dim=1)
                    traj_w = weight_trans(model_b_mn(inp))
                    v = torch.zeros(n_sample, 2).cuda()
                    traj_id = torch.zeros(n_sample, M).cuda()
                    for n_ind in range(n_sample):
                        traj_pick = np.random.choice(np.arange(M), p=traj_w[n_ind, :].detach().numpy())
                        traj_id[n_ind, traj_pick] = 1
                    for m in range(M):
                        # v = v + torch.diag(traj_w[:, m]) @ model_b_list[m](inp)
                        v = v + torch.diag(traj_id[:, m]) @ model_b_list[m](inp)
                    e = me.sample([n_sample])
                    score = score_est_gpu(x, x, h=h)
                    x = x + (v + s1 * score) * dt + np.sqrt(dt) * e
                    x_track_temp.append(x.cpu().detach().numpy())
                x_target_track.append(x_track_temp)
            else:
                x = torch.tensor(x1.sample(n_sample, replace=True)[['x']].to_numpy()).cuda()
                x_track_temp = []
                x_track_temp.append(x.cpu().detach().numpy())
                for t_ind in range(nt - 1):
                    ti = T - t_grid[-(t_ind + 1)]
                    tf = T - t_grid[-(t_ind + 2)]
                    dt = (tf - ti) / T
                    inp = torch.cat([x, ti * torch.ones(n_sample, 1).cuda() / T], dim=1)
                    traj_w = weight_trans(model_b_mn(inp))
                    v = torch.zeros(n_sample, 2).cuda()
                    traj_id = torch.zeros(n_sample, M).cuda()
                    for n_ind in range(n_sample):
                        traj_pick = np.random.choice(np.arange(M), p=traj_w[n_ind, :].detach().numpy())
                        traj_id[n_ind, traj_pick] = 1
                    for m in range(M):
                        # v = v + torch.diag(traj_w[:, m]) @ model_b_list[m](inp)
                        v = v + torch.diag(traj_id[:, m]) @ model_b_list[m](inp)
                    e = me.sample([n_sample])
                    x = x + v * dt + np.sqrt(dt) * e
                    x_track_temp.append(x.cpu().detach().numpy())
                x_target_track.append(x_track_temp[::-1])
                x_start = x.cpu().detach().numpy()

            obj_b.append(obj)

    return {'model_f': model_f_list,
            'model_b': model_b_list,
            'optimizer_f': optimizer_f_list,
            'optimizer_b': optimizer_b_list,
            'model_f_mn': model_f_mn,
            'model_b_mn': model_b_mn,
            'optimizer_f_mn': optimizer_f_mn,
            'optimizer_b_mn': optimizer_b_mn,
            'cost_f': obj_f,
            'cost_b': obj_b}


def sim_path_ot(res, x0, T, nt=100, t_check=None, s1=1, s2=1, plot=False, use_gpu=False, **_):

    t_data = res['t_data']
    t_grid = np.linspace(0, T, nt)
    if t_check is None:
        t_grid = np.unique(np.concatenate((t_data, t_grid), axis=None))
    else:
        t_grid = np.unique(np.concatenate((t_data, t_grid, t_check), axis=None))
    t_grid.sort()
    nt_grid = t_grid.shape[0]
    nt_data = t_data.shape[0]
    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.tensor(np.diag([s1, s2])).float())
    data = x0
    x = torch.tensor(x0)
    ts = [0]
    n_sample = x.shape[0]
    ind_check = 1
    check = True
    model = res['model']
    if use_gpu:
        model = model.cpu()
    start_loc = res['x0']
    end_loc = res['x1']
    for t_ind in range(nt_grid - 1):
        ti = t_grid[t_ind]
        tf = t_grid[t_ind + 1]
        dt = (tf - ti) / T
        inp = torch.cat([x, ti * torch.ones(n_sample, 1) / T], dim=1)
        e = me.sample([n_sample])
        if check:
            if ind_check < nt_data:
                if ti == t_data[ind_check - 1]:
                    tc0 = t_data[ind_check - 1]
                    tc1 = t_data[ind_check]
                    n_ref = end_loc[ind_check - 1].shape[0]
                    start = torch.tensor(start_loc[ind_check - 1])
                    end = torch.tensor(end_loc[ind_check - 1])
                    ind_check += 1
            else:
                check = False
            gamma = (tf - tc0) / (tc1 - tc0)
            if gamma < 0 or gamma > 1:
                raise ArithmeticError('encountered wrong gamma')
            x_start = (1 - gamma) * start + gamma * end
            cdist = tensor_cost(x, x_start)
            end_ind = cdist.argmin(axis=1)
            # kern_weight = kernel_weight(x, x_start)
            # kern_weight = kern_weight.detach().numpy()
            # end_ind = []
            # for j in range(n_sample):
            #     end_ind.append(np.random.choice(np.arange(n_ref), p=kern_weight[j, :]))
            # x_end = end[end_ind]
            # x_end = kern_weight @ end
            v_free = model(inp)
            v_ref = (x_start[end_ind] - x) / dt - v_free
            v = v_free + gamma * v_ref
        else:
            v = model(inp)
        x = x + v * dt + np.sqrt(dt) * e
        data = np.vstack((data, x.detach().numpy()))
        ts.append(tf)
    data = pd.DataFrame(data, columns=['x', 'y'])
    data['time'] = np.repeat(ts, n_sample)
    if plot:
        data.plot.scatter(x='x', y='y', c='time', s=1, cmap='Spectral', figsize=(10, 8))
    return data


def sim_path_force(model, x0, T, data_full, t_check=None, nt=100, s1=1, s2=1, plot=False, use_gpu=False):

    if use_gpu:
        model = model.cpu()
    t_grid = np.linspace(0, T, nt)
    if t_check is None:
        t_grid = np.unique(np.concatenate((t_check, t_grid), axis=None))
    t_grid.sort()
    t_diff = np.diff(t_grid) / T
    nt_grid = t_grid.shape[0]
    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.tensor(np.diag([s1, s2])).float())
    data = x0
    x = torch.tensor(x0)
    ts = [0]
    n_sample = x.shape[0]
    ind_check = 0
    check = True
    for t_ind in range(nt_grid - 1):
        t = t_grid[t_ind]
        e = me.sample([n_sample])
        inp = torch.cat([x, t * torch.ones(n_sample, 1) / T], dim=1)
        dt = t_diff[t_ind]
        tf = t_grid[t_ind + 1]
        if check:
            tc = t_check[ind_check]
            x_check = torch.tensor(data_full[data_full.time == tc].iloc[:, [0, 1]].to_numpy())
            gamma = (tf - t) / (t_check[ind_check] - t)
            if gamma < 0 or gamma > 1:
                raise ArithmeticError('encountered wrong gamma')
            d1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
            d2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
            c = d1.pow(2) + d2.pow(2)
            ind_close = c.argmin(axis=1)
            dest_ind = np.zeros(n_sample)
            for i in range(n_sample):
                if np.random.uniform() > (1 - 0.1 * gamma):
                    dest_ind[i] = np.random.choice(np.arange(x_check.shape[0]))
                else:
                    dest_ind[i] = ind_close[i]
            x_pred = x_check[dest_ind]
            v_pred = (x_pred - x) / dt
            v = model(inp) * (1 - gamma) + v_pred * gamma
            if tf == t_check[ind_check]:
                if tf == t_check[-1]:
                    check = False
                ind_check += 1
                x = x + v * dt + np.sqrt(dt) * e
            else:
                x = x + v * dt + np.sqrt(dt) * e
                # x = x + v * dt
        else:
            v = model(inp)
            e = me.sample([n_sample])
            x = x + v * dt + np.sqrt(dt) * e
            # x = x + v * dt
        data = np.vstack((data, x.detach().numpy()))
        ts.append(tf)
    data = pd.DataFrame(data, columns=['x', 'y'])
    data['time'] = np.repeat(ts, n_sample)
    if plot:
        data.plot.scatter(x='x', y='y', c='time', s=1, cmap='Spectral', figsize=(10, 8))
    return data


def sim_path_soft(model, x0, T, t_check=None, nt=100, s1=1, s2=1, plot=False, use_gpu=False, **_):

    if use_gpu:
        model = model.cpu()
    t_grid = np.linspace(0, T, nt)
    if t_check is not None:
        t_grid = np.unique(np.concatenate((t_check, t_grid), axis=None))
    t_diff = np.diff(t_grid) / T
    nt_grid = t_grid.shape[0]
    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.tensor(np.diag([s1, s2])).float())
    data = x0
    x = torch.tensor(x0)
    ts = [0]
    n_sample = x.shape[0]
    for t_ind in range(nt_grid - 1):
        t = t_grid[t_ind]
        dt = t_diff[t_ind]
        inp = torch.cat([x, t * torch.ones(n_sample, 1) / T], dim=1)
        v = model(inp)
        e = me.sample([n_sample])
        x = x + v * dt + np.sqrt(dt) * e
        # x = x + v * dt
        data = np.vstack((data, x.detach().numpy()))
        ts.append(t_grid[t_ind + 1])
    data = pd.DataFrame(data, columns=['x', 'y'])
    data['time'] = np.repeat(ts, n_sample)
    if plot:
        data.plot.scatter(x='x', y='y', c='time', s=1, cmap='Spectral', figsize=(10, 8))
    return data


def sim_path_soft_seg(model, x0, T, bound, t_check=None, nt=100, s1=1, s2=1, plot=False, use_gpu=False, **_):

    if use_gpu:
        for i_m in range(len(model)):
            model[i_m] = model[i_m].cpu()
    # t_grid = np.linspace(0, T, nt)
    t_grid = np.concatenate((np.linspace(0, T, nt), bound[1:, 0]), axis=None)
    if t_check is not None:
        t_grid = np.unique(np.concatenate((t_check, t_grid), axis=None))
    t_grid.sort()
    t_diff = np.diff(t_grid) / T
    nt_grid = t_grid.shape[0]
    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.tensor(np.diag([s1, s2])).float())
    data = x0
    x = torch.tensor(x0)
    ts = [0]
    n_sample = x.shape[0]
    n_seg = len(model)
    for t_ind in range(nt_grid - 1):
        t = t_grid[t_ind]
        dt = t_diff[t_ind]
        inp = torch.cat([x, t * torch.ones(n_sample, 1) / T], dim=1)
        ns = 0
        while ns < n_seg:
            if t >= bound[ns, 0] and t < bound[ns, 1]:
                mod_ind = ns
                ns = n_seg
            ns += 1
        v = model[mod_ind](inp)
        e = me.sample([n_sample])
        x = x + v * dt + np.sqrt(dt) * e
        # x = x + v * dt
        data = np.vstack((data, x.detach().numpy()))
        ts.append(t_grid[t_ind + 1])
    data = pd.DataFrame(data, columns=['x', 'y'])
    data['time'] = np.repeat(ts, n_sample)
    if plot:
        data.plot.scatter(x='x', y='y', c='time', s=1, cmap='Spectral', figsize=(10, 8))
    return data


def sim_path_fb_ot(framework, x0, t_check=None, nt=100, s1=1, s2=1, h=None, plot=False, use_gpu=False, **_):

    model_f = framework['model_f']
    model_b = framework['model_b']
    t_data = framework['t_data']
    tmap_norm = framework['tmap_norm']
    start_loc = framework['x0']
    end_loc = framework['x1']
    T = np.max(t_data)

    if use_gpu:
        model_f = model_f.cpu()
        model_b = model_b.cpu()

    t_grid = np.unique(np.concatenate((np.linspace(0, T, nt), t_data), axis=None))
    if t_check is not None:
        t_grid = np.unique(np.concatenate((t_check, t_grid), axis=None))
    t_grid.sort()
    nt_grid = t_grid.shape[0]
    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.tensor(np.diag([s1, s2])).float())
    data = x0.copy()
    x = torch.tensor(x0)
    ts = [0]
    n_sample = x.shape[0]
    mod_ind = 0
    ind_check = 0
    for t_ind in range(nt_grid - 1):
        ti = t_grid[t_ind]
        tf = t_grid[t_ind + 1]
        if ti == t_data[ind_check]:
            t0_ind = t_ind
            data_temp_x = np.zeros((0, 2))
            t_start = t_data[ind_check]
            t_end = t_data[ind_check + 1]
            start_ref = start_loc[ind_check]
            end_ref = end_loc[ind_check]
            n_start = start_ref.shape[0]
            n_end = end_ref.shape[0]
            start = torch.tensor(start_ref)
            end = np.zeros(start.shape)
            for j in range(n_start):
                dest = np.random.choice(np.arange(n_end),p=tmap_norm[ind_check][j, :])
                end[j, :] = end_ref[dest, :]
            end = torch.tensor(end)
            cdist = tensor_cost(x, start)
            end_ind = cdist.argmin(axis=1)
            new_start = end[end_ind]
            # kw = kernel_weight(x, start, h=h)
            # new_start = kw @ end
            ind_check += 1
        dt = (tf - ti) / (t_end - t_start)
        inp = torch.cat([x, (ti - t_start) * torch.ones(n_sample, 1) / (t_end - t_start)], dim=1)
        v = model_f[mod_ind](inp)
        e = me.sample([n_sample])
        x = x + v * dt + np.sqrt(dt) * e
        data_temp_x = np.vstack((data_temp_x, x.detach().numpy()))
        ts.append(tf)
        if tf == t_end:
            t1_ind = t_ind + 1
            x = new_start
            y = new_start
            data_temp_y = np.zeros((0, 2))
            for ty in np.arange(t1_ind, t0_ind, -1):
                data_temp_y = np.vstack((y.detach().numpy(), data_temp_y))
                ti_y = t_grid[ty]
                tf_y = t_grid[ty - 1]
                dt_y = (ti_y - tf_y) / (t_end - t_start)
                inp_y = torch.cat([y, (ti_y - t_start) * torch.ones(n_sample, 1) / (t_end - t_start)], dim=1)
                v_y = model_b[mod_ind](inp_y)
                e = me.sample([n_sample])
                y = y + v_y * dt_y + np.sqrt(dt_y) * e
            gamma = np.repeat(t_grid[(t0_ind + 1):(t1_ind + 1)], n_sample)
            gamma = (gamma - t_start) / (t_end - t_start)
            data_temp = np.zeros(data_temp_x.shape)
            data_temp[:, 0] = (1 - gamma) * data_temp_x[:, 0] + gamma * data_temp_y[:, 0]
            data_temp[:, 1] = (1 - gamma) * data_temp_x[:, 1] + gamma * data_temp_y[:, 1]
            data = np.vstack((data, data_temp))
            mod_ind += 1
    data = pd.DataFrame(data, columns=['x', 'y'])
    data['time'] = np.repeat(ts, n_sample)
    if plot:
        data.plot.scatter(x='x', y='y', c='time', s=1, cmap='Spectral', figsize=(10, 8))
    return data


def sim_path_mixed(res, x0, T, nt=100, t_check=None, s1=1, s2=1, fb=False, plot=False, use_gpu=False, **_):

    t_grid = np.linspace(0, T, nt)
    if t_check is not None:
        t_grid = np.unique(np.concatenate((t_check, t_grid), axis=None))
    t_diff = np.diff(t_grid) / T
    nt_grid = t_grid.shape[0]
    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.tensor(np.diag([s1, s2])).float())
    weight_trans = nn.Softmin(dim=1)
    data = x0
    x = torch.tensor(x0)
    ts = [0]
    n_sample = x.shape[0]
    if fb:
        model_list = res['model_f']
        model_mn = res['model_f_mn']
    else:
        model_list = res['model_drift']
        model_mn = res['model_mn']
    M = len(model_list)
    if use_gpu:
        for m in range(M):
            model_list[m] = model_list[m].cpu()
            model_mn = model_mn.cpu()
    for t_ind in range(nt_grid - 1):
        t = t_grid[t_ind]
        dt = t_diff[t_ind]
        inp = torch.cat([x, t * torch.ones(n_sample, 1) / T], dim=1)
        traj_w = weight_trans(model_mn(inp))
        v = torch.zeros(n_sample, 2)
        for m in range(M):
            v = v + torch.diag(traj_w[:, m]) @ model_list[m](inp)
        e = me.sample([n_sample])
        x = x + v * dt + np.sqrt(dt) * e
        # x = x + v * dt
        data = np.vstack((data, x.detach().numpy()))
        ts.append(t_grid[t_ind + 1])
    data = pd.DataFrame(data, columns=['x', 'y'])
    data['time'] = np.repeat(ts, n_sample)
    if plot:
        data.plot.scatter(x='x', y='y', c='time', s=1, cmap='Spectral', figsize=(10, 8))
    return data


class NeuralNetwork(nn.Module):

    def __init__(self, d_in, d_out, d_hid, n_layers=2):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = []
        layers.append(nn.Linear(d_in, d_hid))
        layers.append(nn.ReLU())
        for n in range(n_layers):
            layers.append(nn.Linear(d_hid, d_hid))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(d_hid, d_out))
        self.flow = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.flow(x.float())


class NeuralNetwork2D(nn.Module):

    def __init__(self, d_in, d_out, d_hid, n_layers=20):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(d_in, d_hid, kernel_size=3, padding='same'))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(d_hid))
        for n in range(n_layers):
            layers.append(nn.Conv2d(d_hid, d_hid, 3, padding='same'))
            layers.append(nn.ReLU())
            if n % 4 == 0 and n > 0:
                layers.append(nn.BatchNorm2d(d_hid))
        layers.append(nn.Conv2d(d_hid, d_out, kernel_size=3, padding='same'))
        # layers.append(nn.Conv2d(d_hid, d_out, kernel_size=3, padding='valid'))
        # layers.append(nn.BatchNorm2d(d_out))
        self.flow = nn.Sequential(*layers)

    def forward(self, x):
        return self.flow(x.float())


class NeuralVol2D(nn.Module):

    def __init__(self, d_in, d_out, d_hid, n_layers=20):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(d_in, d_hid, kernel_size=3, padding='same'))
        layers.append(nn.ReLU())
        # layers.append(nn.BatchNorm2d(d_hid))
        for n in range(n_layers):
            layers.append(nn.Conv2d(d_hid, d_hid, 3, padding='same'))
            layers.append(nn.ReLU())
            # if n % 4 == 0:
            #     layers.append(nn.BatchNorm2d(d_hid))
        layers.append(nn.Conv2d(d_hid, d_out, kernel_size=3, padding='same'))
        # layers.append(nn.Sigmoid())
        self.flow = nn.Sequential(*layers)

    def forward(self, x):
        return self.flow(x.float())


class View(nn.Module):

    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class NeuralEncoder(nn.Module):

    def __init__(self, d_in, d_out, d_hid, n_layers=20):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(d_in, d_hid, kernel_size=8, padding='same'))
        layers.append(nn.ReLU())
        # layers.append(nn.BatchNorm2d(d_hid))
        for n in range(n_layers):
            layers.append(nn.Conv2d(d_hid, d_hid, 8, padding='same'))
            layers.append(nn.ReLU())
            # if n % 4 == 0:
            #     layers.append(nn.BatchNorm2d(d_hid))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(d_hid, d_out))
        # layers.append(nn.Sigmoid())
        self.flow = nn.Sequential(*layers)

    def forward(self, x):
        return self.flow(x.float())


class NeuralDecoder(nn.Module):

    def __init__(self, d_in, d_out, d_hid, height, width, n_layers=20):
        super().__init__()
        layers = []
        layers.append(nn.Linear(d_in, d_hid * height * width))
        layers.append(View((d_hid, height, width)))
        layers.append(nn.ReLU())
        # layers.append(nn.BatchNorm2d(d_hid))
        for n in range(n_layers):
            layers.append(nn.Conv2d(d_hid, d_hid, 8, padding='same'))
            layers.append(nn.ReLU())
            # if n % 4 == 0:
            #     layers.append(nn.BatchNorm2d(d_hid))
        layers.append(nn.Conv2d(d_hid, d_out, kernel_size=8, padding='same'))
        self.flow = nn.Sequential(*layers)

    def forward(self, x):
        return self.flow(x.float())


class NeuralLinear(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.flatten = nn.Flatten()
        self.flow = nn.Sequential(nn.Linear(d_in, d_out, bias=False))

    def forward(self, x):
        x = self.flatten(x)
        return self.flow(x.float())


class NeuralFeat(nn.Module):

    def __init__(self, d_in, d_out, d_hid, n_val_layers=20, n_same_layers=20):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(d_in, d_hid, kernel_size=3, padding='same'))
        layers.append(nn.ReLU())
        for n in range(n_val_layers):
            layers.append(nn.Conv2d(d_hid, d_hid, kernel_size=3, padding='valid'))
            layers.append(nn.ReLU())
            if n % 4 == 0 and n > 0:
                layers.append(nn.BatchNorm2d(d_hid))
        for n in range(n_same_layers):
            layers.append(nn.Conv2d(d_hid, d_hid, kernel_size=3, padding='same'))
            layers.append(nn.ReLU())
            if n % 4 == 0 and n > 0:
                layers.append(nn.BatchNorm2d(d_hid))
        layers.append(nn.Conv2d(d_hid, d_out, kernel_size=3, padding='same'))
        self.flow = nn.Sequential(*layers)

    def forward(self, x):
        return self.flow(x.float())


class NNconv5_3(nn.Module):

    def __init__(self, d_in, d_out, d_hid, size=3):
        super().__init__()
        self.flow = nn.Sequential(nn.Conv2d(d_in, d_hid, kernel_size=size, padding='same'),
                                  nn.ReLU(),
                                  nn.Conv2d(d_hid, d_hid, kernel_size=size, padding='same'), # conv1
                                  nn.ReLU(),
                                  nn.Conv2d(d_hid, d_hid, kernel_size=size, padding='same'), # conv2
                                  nn.ReLU(),
                                  nn.Conv2d(d_hid, d_hid, kernel_size=size, padding='same'), # conv3
                                  nn.ReLU(),
                                  nn.Conv2d(d_hid, d_hid, kernel_size=size, padding='same'), # conv4
                                  nn.ReLU(),
                                  nn.Conv2d(d_hid, d_out, kernel_size=size, padding='same')) # conv5

    def forward(self, x):
        return self.flow(x.float())