import numpy as np
import pandas as pd

param_names = [# regularizer
               'r_v', 'r_ent', 'r_ent_v', 'r_kl', 'r_lock',
               'reg', 'reg1', 'reg2', 'k', 'lock_dist',
               # model setting
               'nt_grid', 'n_seg', 'n_sample', 'nt_subgrid', 'n_mixed', 'fb_iter',
               # simulation setting
               'nt', 'n_test', 's1', 's2', 'h'
               # optimization
               'lr', 'n_iter',
               # mc
               'M'
               # setting id
               'setting_id']

param_info = {#regularizer
              'r_v': float,
              'r_ent': float,
              'r_ent_v' : float,
              'r_kl': float,
              'r_lock': float,
              'reg': float,
              'reg1': float,
              'reg2': float,
              'k': int,
              'lock_dist': float,
              # model setting
              'nt_grid': int,
              'n_seg': int,
              'n_sample': int,
              'nt_subgrid': int,
              'n_mixed': int,
              'fb_iter': int,
              # simulation setting
              'nt': int,
              'n_test': int,
              's1': float,
              's2': float,
              'h': float,
              # optimization
              'lr': float,
              'n_iter': int,
              # mc
              'M': int,
              # setting id
              'setting_id': int}

param_multi_setting = pd.DataFrame(columns=param_info.keys())

# default setting
default = {#regularizer
           'r_v': 1,
           'r_ent': 0.01,
           'r_ent_v' : 1,
           'r_kl': 0.001,
           'r_lock': 1,
           'reg': 0.01,
           'reg1': 50,
           'reg2': 50,
           'k': 5,
           'lock_dist': 0.001,
           # model setting
           'nt_grid': 250,
           'n_seg': 5,
           'n_sample': 100,
           'nt_subgrid': 10,
           'n_mixed': 8,
           'fb_iter': 5,
           # simulation setting
           'nt': 250,
           'n_test': 1000,
           's1': 0.005,
           's2': 0.005,
           'h': 1,
           # optimization
           'lr': 0.001,
           'n_iter': 100,
           # mc
           'M': 20,
           # setting id
           'setting_id': 0}

param_multi_setting = pd.concat([param_multi_setting, pd.DataFrame([default])], ignore_index=True)

r_v_list = [0.1, 1, 10]
r_ent_list = [0.1, 1, 10]
r_ent_v_list = [0.1, 1, 10]
r_kl_list = [0.1, 1, 10]
r_lock_list = [0.1, 1, 10]
lock_dist_list = [0.001, 0.01, 0.1, 1]
k_list = [3, 5, 10]
n_mixed_list = [4, 8, 16]
fb_iter_list = [5, 10, 20]

setting_id = 1
for ent in r_ent_list:
    for kl in r_kl_list:
        for lock in r_lock_list:
            for m_traj in n_mixed_list:
                for fb_iter in fb_iter_list:
                    setting = default.copy()
                    setting['r_ent'] = ent
                    setting['r_ent_v'] = ent
                    setting['r_kl'] = kl
                    setting['r_lock'] = lock
                    setting['n_mixed'] = m_traj
                    setting['fb_iter'] = fb_iter
                    setting['setting_id'] = setting_id
                    setting_id += 1
                    param_multi_setting = pd.concat([param_multi_setting, pd.DataFrame([setting])], ignore_index=True)

for name, info in param_info.items():
    param_multi_setting[name] = param_multi_setting[name].astype(info)

param_multi_setting.to_csv('data/param_multi_setting.csv')
