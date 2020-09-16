# %%
# load in baic libraries
import autograd.numpy as np
from autograd import grad
import numpy
from random import random

import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd

import sys

sys.path.append('./')
datapath = './'

from functions import plot_functions

%reload_ext autoreload
%autoreload 2

%matplotlib inline

# %%
# %%
# load in data
csvname = datapath + '2d_classification_data_v1_entropy.csv'
data = np.asarray(pd.read_csv(csvname,header = None))

x = data[0][:,np.newaxis]
y = np.asarray(data[1][:,np.newaxis])
x_b = np.concatenate((np.ones((x.size, 1)), x), axis=1)

# %%
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def sigmoid_least_squares2(w):
    p = y.size
    return (1 / p) * np.sum(((sigmoid(x_b.dot(w)) - y) ** 2))


# sigmoid non-convex logistic least squares cost function
def sigmoid_least_squares(w):
    cost = 0
    for p in range(y.size):
        x_p = x_b[p,:]
        y_p = y[p,:]
        cost += (sigmoid(w[0] + w[1]*x_p[1]) - y_p)**2
    return cost/y.size

def gradient_descent_sigmoid_least_squares(x,y,w,alpha=0.1,max_its=100):    
    # gradient function
    gradient = grad(sigmoid_least_squares)

    # run the gradient descent loop
    weight_history = [w]           # container for weight history
    cost_history = [sigmoid_least_squares(w)]          # container for corresponding cost function history
    for _ in range(max_its):
        # evaluate the gradient, store current weights and cost function value
        grad_eval = gradient(w)
        #grad_eval = (x_b.T.dot(np.exp(-x_b.dot(w))).dot(((1 + np.exp(-x_b.dot(w)))**-2).T).dot((sigmoid(x_b.dot(w))-y))) * 2 / y.size

        grad_norm = np.linalg.norm(grad_eval)
        if grad_norm == 0:
            grad_norm += 10**-6*np.sign(2*np.random.rand(1) - 1)
        grad_eval /= grad_norm
        
        # take gradient descent step
        w = w - alpha*grad_eval
        
        # record weight and cost
        weight_history.append(w)
        cost_history.append(sigmoid_least_squares(w))
    return weight_history,cost_history

# %%
w = np.asarray([20.0,-20.0])[:,np.newaxis]
w_h,c_h = gradient_descent_sigmoid_least_squares(x_b,y,w, 1, 6)
# %%
x_f = np.linspace(-1, 5, 100)[:,np.newaxis]
x_fb = np.concatenate((np.ones((x_f.size, 1)), x_f), axis=1)
plt.plot(x_f, sigmoid(x_fb.dot(w_h[-1])))
plt.scatter(x,y)
  # %%
# %%
gs = gridspec.GridSpec(1, 1, width_ratios=[1]) 
ax = plt.subplot(gs[0])
plot_functions.contour_plot(sigmoid_least_squares, ax, 31, 25)
plot_functions.weight_history_plot(ax,sigmoid_least_squares,w_h)
