import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn

import ot


def tensor_cost(x, y):
    c1 = x[:, 0].reshape(-1, 1) - y[:, 0]
    c2 = x[:, 1].reshape(-1, 1) - y[:, 1]
    c = c1.pow(2) + c2.pow(2)
    return c

def kernel(x, h=None):
    if h is None:
        h = x.T.cov() * (x.shape[0] ** (-1 / 6))
        # h = x.T.cov() / x.shape[0]
        h = h.detach().numpy()
    h = np.linalg.inv(h)
    c1 = x[:, 0].reshape(-1, 1) - x[:, 0]
    c2 = x[:, 1].reshape(-1, 1) - x[:, 1]
    c = h[0, 0] * c1.pow(2) + (h[0, 1] + h[1, 0]) * c1 * c2 + h[1, 1] * c2.pow(2)
    c = torch.exp(-0.5 * c) / torch.tensor(2 * torch.pi * np.sqrt(h[0, 0] * h[1, 1] - h[0, 1] * h[1, 0]))   
    return c.mean(axis=0)


def kernel_pred(x, y, h=None):
    if h is None:
        h = x.T.cov() * (x.shape[0] ** -(1 / 6))
        # h = x.T.cov() / x.shape[0]
        h = h.detach().numpy()
    h = np.linalg.inv(h)
    c1 = x[:, 0].reshape(-1, 1) - y[:, 0]
    c2 = x[:, 1].reshape(-1, 1) - y[:, 1]
    c = h[0, 0] * c1.pow(2) + (h[0, 1] + h[1, 0]) * c1 * c2 + h[1, 1] * c2.pow(2)
    c = torch.exp(-0.5 * c) / torch.tensor(2 * torch.pi * np.sqrt(h[0, 0] * h[1, 1] - h[0, 1] * h[1, 0]))   
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
        
        
def train_alg_mfc_ot(data, T, lr=0.001,
                     n_sample=100, n_iter=128, nt_grid=100, 
                     error_s1=1, error_s2=1,
                     h=None,
                     r_v=0.01, r_ent=0.1, r_ent_v=1, r_lock=1,
                     reg=1, reg1=1, reg2=1,
                     track=False):
    
    t_data = data.time.unique()
    t_data.sort()
    t_grid = np.unique(np.concatenate((t_data, np.linspace(0, T, nt_grid)), axis=None))
    t_grid.sort()
    nt_data = t_data.shape[0]
    nt_model = t_grid.shape[0]
    n_ref = n_sample * 3
    dirac = np.repeat(1, n_ref) / n_ref
    tmap = []
    start_loc = []
    end_loc = []
    for i in range(nt_data - 1):
        data0 = data[data.time == t_data[i]][['x', 'y']].sample(n_ref, replace=True).to_numpy()
        data1 = data[data.time == t_data[i + 1]][['x', 'y']].sample(n_ref, replace=True).to_numpy()
        costm = ot.compute_dist(data0, data1, dim=2, single=False)
        start_loc.append(data0)
        reg_list = (100 - reg) * np.exp(-np.arange(100)) + reg
        tmap_temp = ot.ot_unbalanced_log_stabilized(dirac, dirac, costm, reg, reg1, reg2, reg_list=reg_list)
        tmap.append(tmap_temp)
        tmap_norm = tmap_temp.copy()
        tmap_norm = np.diag(1 / tmap_norm.sum(axis=1)) @ tmap_norm
        end_loc.append(tmap_norm @ data1)
        # end_loc.append(data1[tmap_norm.argmax(axis=1)])
    
    model = NeuralNetwork(3, 2, 100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)
    
    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), 
                                                                    torch.tensor(np.diag([error_s1, error_s2])).float())
    
    x0 = data[data.time == 0]
    
    x_tensor = []
    
    for t in t_data:
        x_tensor.append(torch.from_numpy(data[data.time == t][['x','y']].to_numpy()))
        
    obj = []
    
    for n in range(n_iter):
        
        l = torch.tensor(0.)
        
        x = torch.from_numpy(x0.sample(n_sample, replace=True)[['x','y']].to_numpy())
        ind_check = 1
            
        check = True
        for t_ind in range(nt_model - 1):
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
                        start = torch.tensor(start_loc[ind_check - 1])
                        end = torch.tensor(end_loc[ind_check - 1])
                        ind_check += 1
                else:
                    check = False
                gamma = (tf - tc0) / (tc1 - tc0)
                if gamma < 0 or gamma > 1:
                    raise ArithmeticError('encountered wrong gamma (gamma = ' + str(gamma) + ')')
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
                l = l + r_lock * (x_start[end_ind] - x).pow(2).sum(axis=1).mean()
            else:
                v = model(inp)
            l = l + r_v * dt * (v.pow(2).sum(axis=1).mean())
            x = x + v * dt + np.sqrt(dt) * e
            phat = kernel(x, h=h)
            pvhat = kernel(v)
            l = l + r_ent * dt * phat.log().mean()
            l = l + r_ent_v * dt * pvhat.log().mean()
        
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


def train_alg_mfc_force(data, T, lr=0.001,
                        n_sample=100, n_iter=128, nt_grid=100, 
                        error_s1=1, error_s2=1,
                        h=None,
                        r_v=0.01, r_ent=0.1, r_kl=1,
                        track=False):
    
    t_data = data.time.unique()
    t_data.sort()
    t_grid = np.unique(np.concatenate((t_data, np.linspace(0, T, nt_grid)), axis=None))
    t_grid.sort()
    nt = t_grid.shape[0]
        
    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), 
                                                                    torch.tensor(np.diag([error_s1, error_s2])).float())
        
    model = NeuralNetwork(3, 2, 100)
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


def train_alg_mfc_soft(data, T, lr=0.001,
                       n_sample=100, n_iter=128, nt_grid=100, 
                       error_s1=1, error_s2=1,
                       h=None, k=5, lock_dist=0.01,
                       r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1,
                       track=False):
    
    t_data = data.time.unique()
    t_data.sort()
    t_grid = np.unique(np.concatenate((t_data, np.linspace(0, T, nt_grid)), axis=None))
    t_grid.sort()
    nt = t_grid.shape[0]
        
    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), 
                                                                    torch.tensor(np.diag([error_s1, error_s2])).float())
        
    model = NeuralNetwork(3, 2, 128)
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
            pvhat = kernel(v, h=h)
            l = l + r_ent_v * dt * (pvhat.log().mean())
            if check:
                if tf == t_data[ind_check]:
                    x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample).to_numpy())
                    # x_support = torch.from_numpy(data.sample(1000)[['x', 'y']].to_numpy())
                    c1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
                    c2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
                    c = c1.pow(2) + c2.pow(2)
                    # prop_in = torch.diag(1 / c.sum(axis=1)) @ c
                    p = kernel_pred(x_check, x, h=h)
                    l = l + r_kl * (phat.log() - p.log()).mean()
                    c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                    ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                    l = l + r_lock * (torch.max(c_lowk.sqrt() - lock_dist, torch.tensor(0.))).sum(axis=1).mean()
                    l = l + r_lock * (torch.max(ctr_lowk.sqrt() - lock_dist, torch.tensor(0.))).sum(axis=0).mean()
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


def train_alg_mfc_soft_seg(data, T, lr=0.001,
                           n_sample=100, n_iter=128, nt_grid=100, n_seg=5,
                           error_s1=1, error_s2=1,
                           h=None, k=5, lock_dist=0.01,
                           r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1, 
                           track=False):
    
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
                                                                    torch.tensor(np.diag([error_s1, error_s2])).float())
    
    model = []
    optimizer = []
    obj = []
    
    for i in range(n_seg):
        model.append(NeuralNetwork(3, 2, 64))
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


def train_alg_mfc_fb_ot(data, lr=0.001,
                        n_sample=100, n_iter=128, nt_subgrid=10, 
                        error_s1=1, error_s2=1,
                        h=None, k=5, lock_dist=0.01,
                        r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1,
                        reg=1, reg1=1, reg2=2,
                        track=False):
    
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
                                                                    torch.tensor(np.diag([error_s1, error_s2])).float())
    for i in range(nt_data - 1):
        data0 = data[data.time == t_data[i]][['x', 'y']].sample(n_ref, replace=True).to_numpy()
        data1 = data[data.time == t_data[i + 1]][['x', 'y']].sample(n_ref, replace=True).to_numpy()
        costm = ot.compute_dist(data0, data1, dim=2, single=False)
        start_loc.append(data0)
        end_loc.append(data1)
        reg_list = (100 - reg) * np.exp(-np.arange(100)) + reg
        # tmap_temp = ot.ot_unbalanced_log_stabilized(dirac, dirac, costm, reg, reg1, reg2, reg_list=reg_list)
        tmap_temp = ot.ot_balanced_log_stabilized(dirac, dirac, costm, reg, reg_list=reg_list)
        tmap.append(tmap_temp)
        tmap_norm_temp = tmap_temp.copy()
        tmap_norm_temp = np.diag(1 / tmap_norm_temp.sum(axis=1)) @ tmap_norm_temp
        tmap_norm.append(tmap_norm_temp)
        # end_loc.append(tmap_norm_temp @ data1)
        # end_loc.append(data1[tmap_norm_temp.argmax(axis=1)])
        model_f.append(NeuralNetwork(3, 2, 100))
        model_b.append(NeuralNetwork(3, 2, 100))
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


def train_alg_mfc_mixed(data, T, lr=0.001, M=5,
                        n_sample=100, n_iter=128, nt_grid=100, 
                        error_s1=1, error_s2=1,
                        h=None, k=5, lock_dist=0.01,
                        r_v=0.01, r_ent=0.1, r_kl=1, r_ent_v=1, r_lock=1,
                        track=False):
    
    t_data = data.time.unique()
    t_data.sort()
    t_grid = np.unique(np.concatenate((t_data, np.linspace(0, T, nt_grid)), axis=None))
    t_grid.sort()
    nt = t_grid.shape[0]
        
    me = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), 
                                                                    torch.tensor(np.diag([error_s1, error_s2])).float())
        
    model_list = []
    optimizer_list = []
    for m in range(M):
        model_list.append(NeuralNetwork(3, 2, 128))
        optimizer_list.append(torch.optim.Adam(model_list[m].parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5))
    
    model_mn = NeuralNetwork(3, M, 128)
    optimizer_mn = torch.optim.Admam(model_mn.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)
    
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
            pvhat = kernel(v, h=h)
            l = l + r_ent_v * dt * (pvhat.log().mean())
            if check:
                if tf == t_data[ind_check]:
                    x_check = torch.from_numpy(data[data.time == tf][['x','y']].sample(n_sample).to_numpy())
                    # x_support = torch.from_numpy(data.sample(1000)[['x', 'y']].to_numpy())
                    c1 = x[:, 0].reshape(-1, 1) - x_check[:, 0]
                    c2 = x[:, 1].reshape(-1, 1) - x_check[:, 1]
                    c = c1.pow(2) + c2.pow(2)
                    # prop_in = torch.diag(1 / c.sum(axis=1)) @ c
                    p = kernel_pred(x_check, x, h=h)
                    l = l + r_kl * (phat.log() - p.log()).mean()
                    c_lowk, c_rank = c.topk(k=k, dim=1, largest=False)
                    ctr_lowk, ctr_rank = c.topk(k=k, dim=0, largest=False)
                    l = l + r_lock * (torch.max(c_lowk.sqrt() - lock_dist, torch.tensor(0.))).sum(axis=1).mean()
                    l = l + r_lock * (torch.max(ctr_lowk.sqrt() - lock_dist, torch.tensor(0.))).sum(axis=0).mean()
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


def sim_path_ot(res, x0, T, nt=100, t_check=None, s1=1, s2=1, plot=False):  
    
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


def sim_path_force(model, x0, T, data_full, t_check=None, nt=100, s1=1, s2=1, plot=False):  
        
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


def sim_path_soft(model, x0, T, t_check=None, nt=100, s1=1, s2=1, plot=False):  
        
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
        

def sim_path_soft_seg(model, x0, T, bound, t_check=None, nt=100, s1=1, s2=1, plot=False):  
    
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


def sim_path_fb_ot(framework, x0, t_check=None, nt=100, s1=1, s2=1, h=None, plot=False):
    
    model_f = framework['model_f']
    model_b = framework['model_b']
    t_data = framework['t_data']
    tmap_norm = framework['tmap_norm']
    start_loc = framework['x0']
    end_loc = framework['x1']
    T = np.max(t_data)
    
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

        
class NeuralNetwork(nn.Module):
    
    def __init__(self, d_in, d_out, d_hid):
        super().__init__()
        self.flatten = nn.Flatten()
        self.flow = nn.Sequential(nn.Linear(d_in, d_hid),
                                  nn.ELU(),
                                  nn.Linear(d_hid, d_hid),
                                  nn.ELU(),
                                  nn.Linear(d_hid, d_hid),
                                  nn.ELU(),
                                  nn.Linear(d_hid, d_out))
        
        
    def forward(self, x):
        x = self.flatten(x)
        return self.flow(x.float())
    
