# %%
# load in baic libraries
from functions import plot_functions
import numpy as np
from random import random
import math

import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd

import sys

sys.path.append('./')
datapath = './'


%reload_ext autoreload
%autoreload 2

%matplotlib inline

# %%
# load in data
csvname = datapath + '2d_classification_data_v1_entropy.csv'
data = np.asarray(pd.read_csv(csvname, header=None))

x = data[0][:, np.newaxis]
y = np.asarray(data[1][:, np.newaxis])
x_b = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

# %%


def cross_entropy_predicton(x, w):
    x_b = np.concatenate((np.ones((x.size, 1)), x), axis=1)
    return sigmoid(model(x_b, w))
# compute linear combination of input point


def model(x, w):
    return x.dot(w)

# define sigmoid function


def sigmoid(t):
    return 1/(1 + np.exp(-t))

# the convex cross-entropy cost function


def cross_entroy_cost(w):
    p = y.size
    a = sigmoid(model(x_b, w))

    a[(a == 0.0)] = 10**-6
    a[(1 - a == 0.0)] = 1 - 10**-6

    return - (1 / p) * (np.log(a).T.dot(y) + np.log(1-a).T.dot(1-y))[0][0]

# the gradient of cross-entropy cost function


def cross_entroy_gradient(w):
    p = y.size
    a = sigmoid(x_b.dot(w))
    return - (1 / p) * x_b.T.dot((y - a))


def gradient_descent(x, y, w, cost, gradient, alpha=0.1, max_its=100):
    # run the gradient descent loop
    weight_history = [w]           # container for weight history
    # container for corresponding cost function history
    cost_history = [cost(w)]
    for _ in range(max_its):
        # evaluate the gradient, store current weights and cost function value
        grad_eval = gradient(w)

        # take gradient descent step
        w = w - alpha*grad_eval

        # record weight and cost
        weight_history.append(w)
        cost_history.append(cost(w))
    return weight_history, cost_history


# %%
w = np.asarray([3.0, 3.0])[:, np.newaxis]
w_h, c_h = gradient_descent(
    x_b, y, w, cross_entroy_cost, cross_entroy_gradient, 1, 1000)
# %%
w_hat = w_h[-1]
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2], hspace=0.5)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[1, :], projection='3d')

plot_functions.scatter_plot(ax1, cross_entropy_predicton, w_hat, (x, y), -1, 5)

plot_functions.contour_plot(ax2, cross_entroy_cost, 12, 35)
plot_functions.weight_history_plot(ax2, cross_entroy_cost, w_h)

plot_functions.surface_plot(ax3, cross_entroy_cost,
                            xmin=-10, ymax=10, rotation=180)
plot_functions.weight_history_plot(ax3, cross_entroy_cost, w_h)
