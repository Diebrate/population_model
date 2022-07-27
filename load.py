import numpy as np
import pyreadr
import pandas as pd

def load(data_name, frac=1, **kwargs):
    if data_name == 'root':
        # root data
        data_all = pyreadr.read_r('data/data.rds')
        data_all = pd.DataFrame(data_all[None])
        data_all.columns = ['x', 'y', 'time']
    elif data_name == 'root_syn':
        data_all = pd.read_csv('data/df_root_syn.csv')
        data_all = data_all[['x', 'y', 'time']]
    elif data_name == 'wot':
        # wot data
        data_all = pd.read_csv('data/df_wot.csv')
        data_all = data_all[['x', 'y', 'day']]
        data_all.columns = ['x', 'y', 'time']
        data_all[['x']] /= 10000
        data_all[['y']] /= 10000
    elif data_name == 'syn':
        # synthetic data
        cov = np.array([[1, 0], [0, 1]])
        start = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=5000)
        end1 = np.random.multivariate_normal(mean=[10, 10], cov=cov, size=2500)
        end2 = np.random.multivariate_normal(mean=[10, -10], cov=cov, size=2500)
        data_all = pd.DataFrame(np.vstack((start, end1, end2)))
        data_all.columns = ['x', 'y']
        data_all['time'] = np.repeat([0, 1], 5000)
    elif data_name == 'circle':
        # point to ring data
        cov = np.array([[1, 0], [0, 1]])
        start = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=5000) * 0.01
        theta = np.random.uniform(low=0, high=2 * np.pi, size=5000)
        end = np.vstack((10 * np.cos(theta), 10 * np.sin(theta))).T
        data_all = pd.DataFrame(np.vstack((start, end)))
        data_all.columns = ['x', 'y']
        data_all['time'] = np.repeat([0, 1], 5000)
    elif data_name == 'spiral':
        # point to ring data
        cov = np.array([[1, 0], [0, 1]])
        sigma = np.sqrt(0.1)
        start = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=5000) * 0.01
        def velo(x1, x2, s=0.5, theta=np.pi * 0.01):
            exp_comp = np.exp(-(np.power(x1, 2) + np.power(x2, 2)) / np.power(s, 2))
            phi = np.array([exp_comp * 2 * x1 / np.power(s, 2) - x1,
                            exp_comp * 2 * x2 / np.power(s, 2) - x2])
            f = 10 * exp_comp * np.array([[np.cos(theta), -np.sin(theta)],
                                          [np.sin(theta), np.cos(theta)]]) @ np.array([[x1],
                                                                                       [x2]])
            return phi + f.flatten()
        n_steps = 10
        tau = 1 / n_steps
        time_points = np.linspace(0, 0.01, n_steps + 1)[1:]
        x_temp = start.copy()
        data_all = start.copy()
        for t in time_points:
            v = np.array([velo(*xs) for xs in x_temp])
            x_temp = x_temp + tau * v + sigma * np.sqrt(tau) * np.random.multivariate_normal(mean=[0, 0], cov=cov, size=5000) * 0.01
            data_all = np.vstack((data_all, x_temp))
        data_all = pd.DataFrame(data_all)
        data_all.columns = ['x', 'y']
        data_all['time'] = np.repeat(np.linspace(0, 0.01, n_steps + 1), 5000)
    elif data_name in [str(i) for i in np.arange(10)]:
        from keras.datasets import mnist
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        value = int(data_name)
        ind = train_y == value
        # start with a point and end with a digit image
        cov = np.array([[1, 0], [0, 1]])
        theta = np.random.uniform(low=0, high=2 * np.pi, size=10000)
        start = np.vstack((20 * np.cos(theta) + 14, 20 * np.sin(theta) + 14)).T
        # start = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=10000) * 1
        end = mnist_scatter(train_X[ind][0])
        data_all = pd.DataFrame(np.vstack((start, end)))
        data_all.columns = ['x', 'y']
        data_all['time'] = np.concatenate((np.repeat(0, 10000), np.repeat(1, end.shape[0])))
    if frac != 1:
        data_all = data_all.sample(frac=0.7, replace=False)

    T = data_all.time.max()

    return data_all, T


def mnist_scatter(img):
    ylim, xlim = img.shape
    data = np.zeros((0, 2))
    for i in range(ylim):
        for j in range(xlim):
            count = int(img[i, j])
            for k in range(count):
                x = np.random.uniform(low=j, high=j + 1)
                y = np.random.uniform(low=ylim - i - 2, high=ylim - i - 1)
                data = np.vstack((data, np.array([x, y])))
    return data