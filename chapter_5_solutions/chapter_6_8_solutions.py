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
csvname = datapath + '3d_classification_data_v0.csv'
data = np.asarray(pd.read_csv(csvname, header=None))

x = data[:-1, :].T
y = data[-1:, :].T
x_b = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

# %%


def softmax_predicton(x, w):
    x_b = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    return np.sign(tanh(model(x_b, w)))
# compute linear combination of input point


def model(x, w):
    return x.dot(w)

# define sigmoid function


def sigmoid(t):
    return 1/(1 + np.exp(-t))

# define tanh function


def tanh(t):
    return 2*sigmoid(t) - 1

# the convex cross-entropy cost function


def softmax_cost(w):
    p = y.size
    a = 1 + np.exp(-y * model(x_b, w))
    return (1 / p) * np.sum(np.log(a))

# the gradient of cross-entropy cost function


def softmax_gradient(w):
    p = y.size
    a = np.exp(y * model(x_b, w))
    return - (1 / p) * (y*x_b).T.dot((1/(1 + a)))


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
g = softmax_cost
w = np.random.randn(3, 1)*30
w_h, c_h = gradient_descent(x_b, y, w, softmax_cost, softmax_gradient, 1, 1000)
w_hat = w_h[-1]
# %%
fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
ax1 = plt.subplot(gs[0, 0], aspect='equal')
ax2 = plt.subplot(gs[0, 1], projection='3d')

plot_functions.decision_boundary_plot(ax1, w_hat, softmax_predicton, (x, y))
plot_functions.decision_boundary_3d_plot(
    ax2, w_hat, softmax_predicton, (x, y), view=[15, -145])


# %%
