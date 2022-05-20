import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ot

data_name = 'root_syn'
# available time fraction 0.1, 0.2, 0.5, 0.8

use_sys = False
if use_sys:
    import sys
    data_name = str(sys.argv[1])

perf = {}
eval_method_list = ['mink', 'kernel']

from param import *

np.random.seed(54321)

for time_frac in [0.1, 0.2, 0.5, 0.8]:
    
    perf_t = {}
    
    for eval_method in eval_method_list:
        
        perf_t[eval_method] = np.empty((2, M), dtype=object)
    
        for m in range(M):
            
            perf_t[eval_method][0, m] = []
            perf_t[eval_method][1, m] = []
            
            df_sim_name = 'data/sim/m' + str(m) + '_' + data_name + '_sim_r' + str(time_frac).replace('.', '_') + '.csv'
            df_test_name = 'data/sim/m' + str(m) + '_' + data_name + '_test_r' + str(time_frac).replace('.', '_') + '.csv'
            df_train_name = 'data/sim/m' + str(m) + '_' + data_name + '_train_r' + str(time_frac).replace('.', '_') + '.csv'
            
            df_sim = pd.read_csv(df_sim_name, index_col=0)
            df_test = pd.read_csv(df_test_name, index_col=0)
            df_train = pd.read_csv(df_train_name, index_col=0)
            
            df_sim_mf = df_sim[df_sim.source == 'mf'].drop('source', axis=1)
            df_sim_ot = df_sim[df_sim.source == 'ot'].drop('source', axis=1)
            
            df_all = pd.concat([df_test, df_train], ignore_index=True)
            
            t_check = df_test.time.unique()
            t_check = t_check[t_check > 0]
            t_check.sort()
        
            for t in t_check:
                x_test_mf = df_sim_mf[df_sim_mf.time == t].drop('time', axis=1).to_numpy()
                x_test_ot = df_sim_ot[df_sim_ot.time == t].drop('time', axis=1).to_numpy()
                x_ref = df_all[df_all.time == t].drop('time', axis=1).sample(np.max([x_test_mf.shape[0], x_test_ot.shape[0]]), replace=True).to_numpy()
                cdist_mf = ot.compute_dist(x_test_mf, x_ref, dim=2, single=False)
                cdist_ot = ot.compute_dist(x_test_ot, x_ref, dim=2, single=False)
                if eval_method == 'mink':
                    cdist_mf.sort(axis=1)
                    cdist_ot.sort(axis=1)
                    cdist_mf_mink = cdist_mf[:, :5]
                    cdist_ot_mink = cdist_ot[:, :5]
                    perf_t[eval_method][0, m].append((cdist_mf_mink < 0.05).mean(axis=1).mean())
                    perf_t[eval_method][1, m].append((cdist_ot_mink < 0.05).mean(axis=1).mean())
                elif eval_method == 'kernel':
                    prob_mf = np.exp(-cdist_mf / 2).mean(axis=0)
                    prob_ot = np.exp(-cdist_ot / 2).mean(axis=0)
                    cdist_ref = ot.compute_dist(x_ref, dim=2)
                    prob_ref = np.exp(-cdist_ref / 2).mean(axis=0)
                    perf_t[eval_method][0, m].append((prob_mf * np.log(prob_mf / prob_ref)).mean())
                    perf_t[eval_method][1, m].append((prob_ot * np.log(prob_ot / prob_ref)).mean())
                print('finished time frac = ' + str(time_frac) + ', eval method ' + eval_method + ', m = ' + str(m) + ', t = ' + str(t))
            perf_t[eval_method][0, m] = np.mean(perf_t[eval_method][0, m])
            perf_t[eval_method][1, m] = np.mean(perf_t[eval_method][1, m])
            
    perf_mean = {em: perf_t[em].astype(float).mean(axis=1).astype(str) for em in eval_method_list}
    perf_var = {em: np.char.add(np.char.add('(', perf_t[em].astype(float).std(axis=1).astype(str)), ')') for em in eval_method_list}
    
    perf_t = {em: np.char.add(perf_mean[em], perf_var[em]) for em in eval_method_list}
    perf['time frac = ' + str(time_frac)] = pd.DataFrame(data=perf_t, index=['mf', 'ot'])
    
perf = pd.concat(perf, axis=1)
    
perf.to_csv('data/sim/' + data_name +'_model_summary.csv')