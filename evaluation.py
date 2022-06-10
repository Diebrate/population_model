import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ot

# available time fraction 0.1, 0.2, 0.5, 0.8
method_list = ['disc_  ot', 'fb_ot', 'mixed']

use_sys = True
if use_sys:
    import sys
    data_name = str(sys.argv[1])
else:
    data_name = 'root'

perf = {}
eval_method_list = ['mink', 'kernel']

from param import *

np.random.seed(54321)

for time_frac in [0.1, 0.2, 0.5, 0.8]:
    
    perf_t = {}
    
    for eval_method in eval_method_list:
        
        perf_t[eval_method] = np.empty((len(method_list), M), dtype=object)
    
        for m in range(M):
            
            for im in range(len(method_list)):
                
                method = method_list[im]
            
                perf_t[eval_method][im, m] = []
                perf_t[eval_method][im, m] = []
            
                df_sim_name = 'data/sim/m' + str(m) + '_' + data_name + '_' + method + '_sim_r' + str(time_frac).replace('.', '_') + '.csv'
                df_test_name = 'data/sim/m' + str(m) + '_' + data_name + '_' + method + '_test_r' + str(time_frac).replace('.', '_') + '.csv'
                df_train_name = 'data/sim/m' + str(m) + '_' + data_name + '_' + method + '_train_r' + str(time_frac).replace('.', '_') + '.csv'
                
                df_sim = pd.read_csv(df_sim_name, index_col=0)
                df_test = pd.read_csv(df_test_name, index_col=0)
                df_train = pd.read_csv(df_train_name, index_col=0)
                
                df_all = pd.concat([df_test, df_train], ignore_index=True)
                df_all = df_all[['x', 'y', 'time']]
                
                t_check = df_test.time.unique()
                t_check = t_check[t_check > 0]
                t_check.sort()
            
                for t in t_check:
                    x_test = df_sim[df_sim.time == t].drop('time', axis=1).to_numpy()
                    x_ref = df_all[df_all.time == t].drop('time', axis=1).sample(x_test.shape[0], replace=True).to_numpy()
                    cdist = ot.compute_dist(x_test, x_ref, dim=2, single=False)
                    if eval_method == 'mink':
                        cdist.sort(axis=1)
                        cdist_mink = cdist[:, :5]
                        perf_t[eval_method][im, m].append((cdist_mink < 0.05).mean(axis=1).mean())
                    elif eval_method == 'kernel':
                        prob = np.exp(-cdist / 2).mean(axis=0)
                        cdist_ref = ot.compute_dist(x_ref, dim=2)
                        prob_ref = np.exp(-cdist_ref / 2).mean(axis=0)
                        perf_t[eval_method][im, m].append((prob * np.log(prob / prob_ref)).mean())
                    print('finished time frac = ' + str(time_frac) + ', eval method ' + eval_method + ', method = ' + method + ', m = ' + str(m) + ', t = ' + str(t))
                perf_t[eval_method][im, m] = np.mean(perf_t[eval_method][im, m])
            
    perf_mean = {em: perf_t[em].astype(float).mean(axis=1).astype(str) for em in eval_method_list}
    perf_var = {em: np.char.add(np.char.add('(', perf_t[em].astype(float).std(axis=1).astype(str)), ')') for em in eval_method_list}
    
    perf_t = {em: np.char.add(perf_mean[em], perf_var[em]) for em in eval_method_list}
    perf['time frac = ' + str(time_frac)] = pd.DataFrame(data=perf_t, index=method_list)
    
perf = pd.concat(perf, axis=1)
    
perf.to_csv('data/sim/' + data_name +'_model_summary.csv')