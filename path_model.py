# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 21:28:45 2022

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/df_own.csv')
data = data[data['day'] != 8.25]
data = data[data['day'] != 8.75]

data.plot.scatter(x='x', y='y', c='day', s=1, cmap='Spectral', figsize=(10, 8))