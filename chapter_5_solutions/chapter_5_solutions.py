# %% [markdown]
'''
# Table of Contents
<p><div class="lev1 toc-item"><a href="#Exercise-5.1.-Fitting-a-regression-line-to-the-student-debt-data" data-toc-modified-id="Exercise-5.1.-Fitting-a-regression-line-to-the-student-debt-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Exercise 5.1. Fitting a regression line to the student debt data</a></div><div class="lev1 toc-item"><a href="#Exercise-5.2.-Kleiber’s-law-and-linear-regression" data-toc-modified-id="Exercise-5.2.-Kleiber’s-law-and-linear-regression-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Exercise 5.2. Kleiber’s law and linear regression</a></div><div class="lev1 toc-item"><a href="#Exercise-5.3.-The-Least-Squares-cost-function-and-a-single-Newton-step" data-toc-modified-id="Exercise-5.3.-The-Least-Squares-cost-function-and-a-single-Newton-step-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Exercise 5.3. The Least Squares cost function and a single Newton step</a></div><div class="lev1 toc-item"><a href="#Exercise-5.4.-Solving-the-normal-equations" data-toc-modified-id="Exercise-5.4.-Solving-the-normal-equations-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Exercise 5.4. Solving the normal equations</a></div><div class="lev1 toc-item"><a href="#Exercise-5.5.-Lipschitz-constant-for-the-Least-Squares-cost" data-toc-modified-id="Exercise-5.5.-Lipschitz-constant-for-the-Least-Squares-cost-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Exercise 5.5. Lipschitz constant for the Least Squares cost</a></div><div class="lev1 toc-item"><a href="#Exercise-5.6.-Compare-the-Least-Squares-and-Least-Absolute-Deviation-costs" data-toc-modified-id="Exercise-5.6.-Compare-the-Least-Squares-and-Least-Absolute-Deviation-costs-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Exercise 5.6. Compare the Least Squares and Least Absolute Deviation costs</a></div><div class="lev1 toc-item"><a href="#Exercise-5.7.-Empirically-confirm-convexity-for-a-toy-dataset" data-toc-modified-id="Exercise-5.7.-Empirically-confirm-convexity-for-a-toy-dataset-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Exercise 5.7. Empirically confirm convexity for a toy dataset</a></div><div class="lev1 toc-item"><a href="#Exercise-5.8.-The-Least-Absolute-Deviations-cost-is-convex" data-toc-modified-id="Exercise-5.8.-The-Least-Absolute-Deviations-cost-is-convex-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Exercise 5.8. The Least Absolute Deviations cost is convex</a></div><div class="lev1 toc-item"><a href="#Exercise-5.9.-Housing-price-and-Automobile-Miles-per-Gallon-prediction" data-toc-modified-id="Exercise-5.9.-Housing-price-and-Automobile-Miles-per-Gallon-prediction-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Exercise 5.9. Housing price and Automobile Miles-per-Gallon prediction</a></div><div class="lev1 toc-item"><a href="#Exercise-5.10.-Improper-tuning-and-weighted-regression" data-toc-modified-id="Exercise-5.10.-Improper-tuning-and-weighted-regression-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Exercise 5.10. Improper tuning and weighted regression</a></div><div class="lev1 toc-item"><a href="#Exercise-5.11.-Multi-output-regression" data-toc-modified-id="Exercise-5.11.-Multi-output-regression-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Exercise 5.11. Multi-output regression</a></div>
'''

# %%
# load in basic libraries
import numpy as np
import jax.numpy as jnp
from jax import grad
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append('./')
datapath = './'

# %% [markdown]
'''
# Exercise 5.1. Fitting a regression line to the student debt data
'''

# %%
# import the dataset
csvname = datapath + 'student_debt_data.csv'
data = np.asarray(pd.read_csv(csvname,header = None))

# extract input
x = data[:,0]
x.shape = (len(x),1)

# pad input with ones
o = np.ones((len(x),1))
x_new = np.concatenate((o,x),axis = 1)

# extract output and re-shape
y = data[:,1]
y.shape = (len(y),1)

# %%
plt.scatter(x, y)

# %% [markdown]
'''
Lets setup the linear system associated to minimizing the Least Squares cost function for this problem and solve it.
'''

# %%
# compute linear combination of input points
def model(x,w):
    y_hat = np .dot(x, w)
    return y_hat

# an implementation of the least squares cost function for linear regression
def least_squares (y,x,w):
    # compute the least squares cost
    cost = np. sum (( model(x, w) - y) ** 2)
    return cost / (2*float(y. size))

# linear descent function - inputs: alpha (steplength parameter), max_its (maximum number of iterations), y (labels), x(fetures), w (initialization)
def linear_gradient_descent(x,y,w,alpha=0.1,max_its=100):    
    # gradient function
    gradient = lambda w : (1 / float(y.size)) * (x.T.dot(x.dot(w) - y))

    # run the gradient descent loop
    weight_history = [w]           # container for weight history
    cost_history = [least_squares(y,x,w)]          # container for corresponding cost function history
    for _ in range(max_its):
        # evaluate the gradient, store current weights and cost function value
        grad_eval = gradient(w)
    
        # take gradient descent step
        w = w - alpha*grad_eval
        
        # record weight and cost
        weight_history.append(w)
        cost_history.append(least_squares(y,x,w))
    return weight_history,cost_history

alpha = 0.0001
max_its = 10
# %%
# import the dataset
csvname = datapath + 'kleibers_law_data.csv'
data = np.loadtxt(csvname,delimiter=',')
x = data[:-1,:]
y = data[-1:,:] 

# log-transform data
x = np.log(x).T
y = np.log(y).T
x = np.concatenate((np.ones((x.shape[0],1)), x), axis=1)

alpha = 0.01
max_its = 10000
w =  np.random.randn(x.shape[1], 1)

# %%
w_h,c_h = linear_gradient_descent(x,y,w,alpha,max_its)
# %%
plt.scatter(range(max_its + 1), c_h)
# %%
# plot data with linear fit - this is optional
s = np.linspace(np.min(x),np.max(x))
w_hat = w_h[-1]
t = w_hat[0] + w_hat[1]*s

figure = plt.figure()
plt.plot(s,t,linewidth = 3,color = 'r')
plt.scatter(x[:,1],y,linewidth = 1,c='k',edgecolor='w')
plt.xlabel('log of mass (in kgs)')
plt.ylabel('log of metabolic rate (in Js)')
# %%
x = np.random.random_sample((400, 1))*(200 - 100) + 100
y =  4 * x - 300 +  20 * np.random.randn(x.size, 1)
x_max = np.abs(x).max(axis=0)

#x = x / x_max

x = np.concatenate((np.ones((x.shape[0],1)), x), axis=1)
alpha = 0.00001
max_its = 10000
w =  np.random.randn(x.shape[1], 1)

w_h,c_h = linear_gradient_descent(x,y,w,alpha,max_its)

# %%
s = np.linspace(np.min(x[:,1]),np.max(x[:,1]))
w_hat = w_h[-1]
t = w_hat[0] + w_hat[1]*s

w_hat2 = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
t2 = w_hat2[0] + w_hat2[1]*s


figure = plt.figure()
plt.plot(s,t,linewidth = 3,color = 'r')
plt.plot(s,t2,linewidth = 3,color = 'b')
plt.scatter(x[:,1],y,linewidth = 1,c='k',edgecolor='w')
print(w_h[-1], w_hat2)

# %%
s = np.linspace(np.min(x[:,1]),np.max(x[:,1]))
w_hat = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
t = w_hat[0] + w_hat[1]*s

figure = plt.figure()
plt.plot(s,t,linewidth = 3,color = 'r')
plt.scatter(x[:,1],y,linewidth = 1,c='k',edgecolor='w')
print(w_hat)

# %%
plt.hist(x[:,1], bins=50)
# %%
