#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


x_dataframe = pd.read_csv('../data/q3/logisticX.csv', header=None, names=['x1', 'x2'], dtype=np.float64)
y_dataframe = pd.read_csv('../data/q3/logisticY.csv', header=None, names=['y'])

X = x_dataframe[['x1', 'x2']].to_numpy()
Y = y_dataframe.to_numpy()

# print(X.shape)
# print(Y.shape)
# print(X)
# print(Y)


# In[3]:


m = X.shape[0] # number of examples
n = X.shape[1] # number of dimensions


# In[4]:


# normalize X
X_mean = np.mean(X, axis=0, dtype=np.float64, keepdims=True)
X_std_dev = np.std(X, axis=0, dtype=np.float64, keepdims=True)
X = (X - X_mean) / X_std_dev
print("X mean:", X_mean)
print("X std. dev.:", X_std_dev, '\n')


# In[5]:


# add intercept term x0=1 to data matrix X 
X = np.concatenate((np.ones(shape=(m, 1), dtype=np.float64), X), axis=1)
# print(X)


# In[6]:


# define log likelihood for logistic function
def LL(theta, X, Y):
    # shapes:    X: m*(n+1)
    #            theta: (n+1)*1
    #            xtheta: m*1
    #            htheta: m*1
    #            Y: m*1 
    xtheta = np.matmul(X, theta)
    htheta = 1.0/(1 + np.exp(-xtheta)) # sigmoid of X@theta
    assert htheta.shape == (m,1) 
    ll = (1.0/m)*np.matmul(Y.T, np.log(htheta)) + (1.0/m)*np.matmul(1-Y.T, np.log(1-htheta))
    return ll.item()


# In[7]:


def derLL(theta, X, Y):
    # shapes:    X: m*(n+1)
    #            theta: (n+1)*1
    #            xtheta: m*1
    #            htheta: m*1
    #            Y: m*1
    #            derLL: (n+1)*1, same as theta
    xtheta = np.matmul(X, theta)
    htheta = 1.0/(1 + np.exp(-xtheta)) # sigmoid of X@theta
    assert htheta.shape == (m,1) 
    der = (1.0/m)*np.matmul(X.T, Y-htheta)
    assert der.shape == (n+1,1) 
    return der    


# In[8]:


def hesLL(theta, X, Y):
    # shapes:    X: m*(n+1)
    #            theta: (n+1)*1
    #            xtheta: m*1
    #            htheta: m*1
    #            Y: m*1
    #            derLL: (n+1)*1, same as theta
    xtheta = np.matmul(X, theta)
    htheta = 1.0/(1 + np.exp(-xtheta)) # sigmoid of X@theta
    assert htheta.shape == (m,1)
    htheta = np.reshape(htheta, newshape=(m,))
    D = -1*(np.diag(htheta))*(np.diag(1-htheta)) # element wise product is matrix multiplication for diagonal matrices
    hes = (1.0/m)*np.matmul(X.T, np.matmul(D, X))
    assert hes.shape == (n+1,n+1) 
    return hes    
    


# In[9]:


def checkHessian(H):
    # check symmetry and negative semidefiniteness (LL function is concave)
    if not np.allclose(H, H.T, rtol=1e-05, atol=1e-08):
        return False
    if not np.all(np.linalg.eigvals(H) <= 0):
        return False
    return True


# In[10]:


def converged(next_ll, ll, error_threshold):
    if abs(next_ll-ll) < error_threshold:
        return True
    return False    
    


# In[11]:


# initialize theta
theta = np.zeros((n+1,1), dtype=np.float64)


# In[12]:


# Newton's Method

t = 1 # iteration number
delta = 1e-12 # log likelihood change threshold for convergence

LL_values = np.empty(shape=[0,], dtype=np.float64)
ll = LL(theta, X, Y)

while True:
    
    print("Iteration", t, ": Log Likelihood =", ll)
    LL_v = np.reshape(ll, newshape=(1,))
    LL_values = np.append(LL_values, LL_v, axis=0)
    
    hessian = hesLL(theta, X, Y)
    assert checkHessian(hessian), "Invalid Hessian"
    
    hessian_inv = np.linalg.inv(hessian)
    
    next_theta = theta - np.matmul(hessian_inv, derLL(theta, X, Y))
    
    next_ll = LL(next_theta, X, Y)
    if converged(next_ll, ll, delta):
        theta = next_theta
        break
        
    t = t+1
    theta = next_theta
    ll = next_ll
    
print("Final Prediction Log Likelihood =", LL(theta, X, Y), '\n')    
print("Optimized Theta:\n", theta, '\n')


# In[13]:


plt.figure(0)
plt.plot(LL_values, label="Log Likelihood", color="b")
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")

plt.legend()
plt.gcf().set_size_inches(6, 5)
plt.savefig('3.png', dpi=100)
plt.title("Log Likelihood vs. Iteration No.")
plt.show()

# In[14]:


def decision_boundary_1(X1, X2, theta):
    return theta[0] + theta[1]*X1 + theta[2]*X2


# In[16]:


plt.figure(1)
plt.scatter(X[:,1:2][Y==0], X[:,2:3][Y==0], marker="o", facecolors='none', edgecolors="b", label="Label 0")
plt.scatter(X[:,1:2][Y==1], X[:,2:3][Y==1], marker="x", color="k", label="Label 1")
# plt.plot(x_values[:,0], np.matmul(x_values, theta)[:,0], label="Hypothesis Function", color="k")
plt.xlabel("x1")
plt.ylabel("x2")

axes = plt.gca()
x1_range = np.linspace(axes.get_xlim()[0]-0.5, axes.get_xlim()[1]+0.5, 100)
x2_range = np.linspace(axes.get_ylim()[0]-0.5, axes.get_ylim()[1]+0.5, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = decision_boundary_1(X1, X2, theta)
plt.contour(X1, X2, Z, levels=[0], cmap="RdGy")

plt.legend()
plt.gcf().set_size_inches(6, 5)
plt.savefig('3b.png', dpi=100)
plt.title("Data Points and Decision Boundary")
plt.show()

