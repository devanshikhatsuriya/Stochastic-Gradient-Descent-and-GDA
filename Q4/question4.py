#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# In[2]:


X_unnorm = np.loadtxt('../data/q4/q4x.dat', unpack = True).T
Y_labels = np.loadtxt('../data/q4/q4y.dat', unpack = True, dtype=str)
Y_labels = np.reshape(Y_labels, newshape=(Y_labels.size, 1))
# print(X_unnorm)
# print(Y_labels)


# In[3]:


m = X_unnorm.shape[0] # number of examples
n = X_unnorm.shape[1] # number of dimensions of features


# In[4]:


X_mean = np.mean(X_unnorm, axis=0, dtype=np.float64, keepdims=True)
X_std_dev = np.std(X_unnorm, axis=0, dtype=np.float64, keepdims=True)
X = (X_unnorm - X_mean) / X_std_dev
print("X mean:", X_mean)
print("X std. dev.:", X_std_dev, '\n')


# In[5]:


# 0 for Alaska, 1 for Canada
Y = np.where(Y_labels=="Alaska", 0, 1)


# In[6]:


phi = np.count_nonzero(Y==1)/m
mu_0 = np.sum((Y==0)*X, axis=0, dtype=np.float64, keepdims=True)/np.count_nonzero(Y==0)
mu_1 = np.sum((Y==1)*X, axis=0, dtype=np.float64, keepdims=True)/np.count_nonzero(Y==1)
sigma = (np.matmul(((Y==0)*(X-mu_0)).T, (Y==0)*(X-mu_0)) + np.matmul(((Y==1)*(X-mu_1)).T, (Y==1)*(X-mu_1))) / m

# shapes     phi: (1,1)
#            mu_0, mu_1: (1,n)
#            sigma: (n,n)

print("\u03C6:", phi)
print("\u03BC\u2080:", mu_0)
print("\u03BC\u2081:", mu_1)
print("\u03A3 = \u03A3\u2080 = \u03A3\u2081:", sigma)


# In[7]:


plt.figure(0)
plt.scatter(X_unnorm[:,0:1][Y==0], X_unnorm[:,1:2][Y==0], marker="o", facecolors='none', label="Alaska", edgecolors="b")
plt.scatter(X_unnorm[:,0:1][Y==1], X_unnorm[:,1:2][Y==1], marker="x", label="Canada", color="k")
plt.xlabel("Diameter in Fresh Water")
plt.ylabel("Density in Marine Water")
plt.legend()
plt.gcf().set_size_inches(6, 5)
plt.savefig('4b.png', dpi=100)
plt.title("Data Points")
plt.show()


# In[8]:


def decision_boundary_1(X1, X2, mu_0, mu_1, sigma_inv):
    f1 = 2*(np.matmul(mu_1-mu_0, sigma_inv)[0,0]*X1 + np.matmul(mu_1-mu_0, sigma_inv)[0,1]*X2)
    f2 = np.matmul(np.matmul(mu_1, sigma_inv), mu_1.T) - np.matmul(np.matmul(mu_0, sigma_inv), mu_0.T)
    return f1-f2


# In[18]:


plt.figure(1)
plt.scatter(X[:,0:1][Y==0], X[:,1:2][Y==0], marker="o", facecolors='none', label="Alaska", edgecolors="b")
plt.scatter(X[:,0:1][Y==1], X[:,1:2][Y==1], marker="x", label="Canada", color="k")
plt.xlabel("Diameter in Fresh Water")
plt.ylabel("Density in Marine Water")

axes = plt.gca()
x1_range = np.linspace(axes.get_xlim()[0]-0.25, axes.get_xlim()[1]+0.25, 100)
x2_range = np.linspace(axes.get_ylim()[0]-0.25, axes.get_ylim()[1]+0.25, 100)
sigma_inv = np.linalg.inv(sigma)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = decision_boundary_1(X1, X2, mu_0, mu_1, sigma_inv)
plt.contour(X1, X2, Z, levels=[0], cmap="RdGy")

plt.legend()
plt.gcf().set_size_inches(6, 5)
plt.savefig('4c.png', dpi=100)
plt.title("Normalized Data and Linear Boundary")
plt.show()



# In[10]:


sigma_0 = np.matmul(((Y==0)*(X-mu_0)).T, (Y==0)*(X-mu_0))/np.count_nonzero(Y==0)
sigma_1 = np.matmul(((Y==1)*(X-mu_1)).T, (Y==1)*(X-mu_1))/np.count_nonzero(Y==1)
print("\u03A3\u2080:", sigma_0)
print("\u03A3\u2081:", sigma_1)


# In[11]:


def decision_boundary_2(X1, X2, mu_0, mu_1, sigma_0_inv, sigma_1_inv, c):
    X = np.stack((X1, X2), axis=2)
    X = np.reshape(X, newshape=(X1.shape[0], X1.shape[1], 2, 1))
    a1 = (X-mu_1.T).transpose(0,1,3,2)
    a2 = (X-mu_0.T).transpose(0,1,3,2)
    f1 = 0.5*(np.matmul(np.matmul(a1, sigma_1_inv), X-mu_1.T))
    f2 = 0.5*(np.matmul(np.matmul(a2, sigma_0_inv), X-mu_0.T))
    return (f1-f2+c).reshape((X1.shape[0], X1.shape[1]))


# In[12]:

plt.figure(2)
plt.scatter(X[:,0:1][Y==0], X[:,1:2][Y==0], marker="o", facecolors='none', label="Alaska", edgecolors="b")
plt.scatter(X[:,0:1][Y==1], X[:,1:2][Y==1], marker="x", label="Canada", color="k")
plt.xlabel("Diameter in Fresh Water")
plt.ylabel("Density in Marine Water")

axes = plt.gca()
x1_range = np.linspace(axes.get_xlim()[0]-0.5, axes.get_xlim()[1]+0.25, 100)
x2_range = np.linspace(axes.get_ylim()[0]-0.25, axes.get_ylim()[1]+0.25, 100)
sigma_0_inv = np.linalg.inv(sigma_0)
sigma_1_inv = np.linalg.inv(sigma_1)
c = np.log(((1-phi)/phi) * np.sqrt(np.linalg.det(sigma_0)) / np.sqrt(np.linalg.det(sigma_1)))
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = decision_boundary_1(X1, X2, mu_0, mu_1, sigma_inv)
plt.contour(X1, X2, Z, levels=[0], cmap="RdGy")
Z = decision_boundary_2(X1, X2, mu_0, mu_1, sigma_0_inv, sigma_1_inv, c)
plt.contour(X1, X2, Z, levels=[0], cmap="RdGy")

plt.legend()
plt.gcf().set_size_inches(6, 5)
plt.savefig('4e.png', dpi=100)
plt.title("Linear and Quadratic Boundaries")
plt.show()





