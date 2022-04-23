#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation


# In[2]:


x_dataframe = pd.read_csv('../data/q1/linearX.csv', header=None, names=['X'], dtype={'X':np.float64})
y_dataframe = pd.read_csv('../data/q1/linearY.csv', header=None, names=['Y'], dtype={'Y':np.float64})
# print("X values: \n", x_dataframe)
# print("Y values: \n", y_dataframe)


# In[3]:


x_values = x_dataframe.to_numpy()
y_values = y_dataframe.to_numpy()
print("X values shape:",x_values.shape)
print("Y values shape:",y_values.shape)
# print("X values:\n", x_values)
# print("Y values:\n", y_values)


# In[4]:


m = x_values.shape[0] # no. of examples
n = x_values.shape[1] # no. of dimensions of each example's x


# In[5]:


# normalize data #
x_mean = np.mean(x_values, axis=0, dtype=np.float64, keepdims=True)
x_std_dev = np.std(x_values, axis=0, dtype=np.float64, keepdims=True)
x_values = (x_values - x_mean) / x_std_dev
print("X mean:", x_mean, "X std. dev.:", x_std_dev)
print("X values shape:",x_values.shape)
# print("X values:\n", x_values)


# In[6]:


# add intercept term x0=1 to data matrix x_values 
x_values = np.hstack((x_values, np.ones(shape=(m, 1), dtype=np.float64)))
# print(x_values)


# In[7]:


# initilaize parameters 
theta = np.zeros(shape=(n+1, 1), dtype=np.float64)
print("Initialized Theta:\n", theta, '\n')


# In[8]:


# define loss function J(theta)
def J(theta, x_values, y_values):
    # shapes:    theta: (n+1)*1
    #            x_values: m*(n+1)          
    #            y_values: m*1 
    difference = y_values - np.matmul(x_values, theta)
    loss = (1/(2*m)) * np.matmul(difference.T, difference)
    return loss.item()   


# In[9]:


# derivative(gradient) of loss J(theta) wrt. theta at given theta
def derJ(theta, x_values, y_values):
    # shapes:    x_values.T: (n+1)*m
    #            y_values: m*1 
    #            theta: (n+1)*1
    #            np.matmul(x_values, theta): m*1 
    #            der: (n+1)*1, same as theta 
    der = (-1/m) * np.matmul(x_values.T, y_values - np.matmul(x_values, theta))
    return der


# In[10]:


# define convergence criteria
def converged(next_loss, loss, error_threshold):
    loss_change = abs(next_loss - loss)
    # print("Absolute Change in Loss:", loss_change, '\n')
    if loss_change < error_threshold:
        return True
    return False       


# In[11]:


# gradient descent
t = 1 # iteration number
eta = 0.001 # learning rate
delta = 1e-11 # loss change threshold for convergence
next_theta = theta

loss_values = np.empty(shape=[0,], dtype=np.float64)
theta_values = np.empty(shape=[0,(n+1),1], dtype=np.float64)

loss = J(theta, x_values, y_values)
print("\tInitially, Loss =", loss)
loss_v = np.reshape(loss, newshape=(1,))
theta_v = np.reshape(theta, newshape=(1,(n+1),1))
loss_values = np.append(loss_values, loss_v, axis=0)
theta_values = np.append(theta_values, theta_v, axis=0)


while True:

    next_theta = theta - eta * derJ(theta, x_values, y_values) 

    loss = J(next_theta, x_values, y_values)
    if t%100==0:
        print("\tIteration", t, ": Loss =", loss)

    loss_v = np.reshape(loss, newshape=(1,))
    theta_v = np.reshape(next_theta, newshape=(1,(n+1),1))
    loss_values = np.append(loss_values, loss_v, axis=0)
    theta_values = np.append(theta_values, theta_v, axis=0) 

    theta = next_theta   

    if converged(loss_values[-1], loss_values[-2], delta):    
        break

    t = t+1

print("\nConverged in", t, "Iterations\n")   
print("Final Prediction Loss =", J(theta, x_values, y_values), '\n')    
print("Optimized Theta:\n", theta, '\n')
# print("Y values vs Predicted Y values:\n", np.hstack((y_values, np.matmul(x_values, theta))))


# In[12]:


plt.figure(0)
plt.scatter(x_values[:,0], y_values[:,0], marker="x", label="Data Points", color="b")
plt.plot(x_values[:,0], np.matmul(x_values, theta)[:,0], label="Hypothesis Function", color="k")
plt.xlabel("Acidity")
plt.ylabel("Density of Wine")
plt.legend()
plt.gcf().set_size_inches(6, 5)
plt.savefig('1b.png', dpi=100)
plt.title("Data and Hypothesis Function")
plt.show()


# In[13]:

# ITERATION PLOT
plt.figure(1)
plt.plot(loss_values, label="Loss", color="b")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss vs. Iteration No.")
plt.legend()
plt.gcf().set_size_inches(6, 5)
plt.savefig('1.png', dpi=100)
plt.show()


# In[14]:

def J_values(theta1, theta0):
    theta_values = np.stack((theta1, theta0), axis=2)
    theta_values = np.reshape(theta_values, newshape=(theta0.shape[0], theta0.shape[1], 2, 1))
    difference = y_values - np.matmul(x_values, theta_values)
    f = (1/(2*m)) * np.matmul(difference.transpose(0,1,3,2), difference)
    return f.reshape((theta0.shape[0], theta0.shape[1])) 


# 3D PLOT
plt.figure(2)
ax = plt.axes(projection='3d')
ax.autoscale(enable=True, axis='both', tight=True)

x1 = np.linspace(-1, 1, 30)
x2 = np.linspace(0, 2, 30)

X1, X2 = np.meshgrid(x1, x2)
Z = J_values(X1, X2)

ax.contour3D(X1, X2, Z, 50, cmap='coolwarm')
ax.plot(theta_values[:,0].reshape((theta_values[:,0].shape[0], )), theta_values[:,1].reshape((theta_values[:,1].shape[0], )), loss_values, 'k-', label = 'Gradient Descent', lw = 1.5)
# ax.scatter(theta_values[:,0], theta_values[:,1], loss_values, marker='o', color='k', s=5, linewidth=0, label="Gradient Descent")

ax.set_xlabel(r'$\theta_1 (slope)$')
ax.set_ylabel(r'$\theta_0 (intercept)$')
ax.set_zlabel(r'$J(\theta)$')

ax.xaxis.set_major_locator(plt.MaxNLocator(6))
ax.yaxis.set_major_locator(plt.MaxNLocator(6))

ax.view_init(45, 30)

plt.legend()
plt.gcf().set_size_inches(6, 5)
plt.savefig('1c.png', dpi=100)
plt.title(r'$J(\theta_0)\;vs.\;\theta$')
plt.show()


# CONTOUR PLOT
plt.figure(3)
plt.xlabel(r'$\theta_1 (slope)$')
plt.ylabel(r'$\theta_0 (intercept)$')

x1_range = np.linspace(-1, 1, 100)
x2_range = np.linspace(0, 2, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = J_values(X1, X2)
plt.contour(X1, X2, Z, levels=25, cmap="RdGy")

plt.plot(theta_values[:,0], theta_values[:,1], linestyle='--', marker='x', color='b', linewidth=1, markersize=5, label="Gradient Descent")

plt.legend()
plt.gcf().set_size_inches(6, 5)
plt.savefig('1d.png', dpi=100)
plt.title("J vs. Theta Contours")
plt.show()


# 3D ANIMATION
fig = plt.figure(4)
ax = plt.axes(projection='3d')
ax.autoscale(enable=True, axis='both', tight=True)
# ax = mplot3d.Axes3D(fig)

x1 = np.linspace(-1, 1, 30)
x2 = np.linspace(0, 2, 30)

X1, X2 = np.meshgrid(x1, x2)
Z = J_values(X1, X2)

ax.contour3D(X1, X2, Z, 50, cmap='coolwarm')

ax.set_xlabel(r'$\theta_1 (slope)$')
ax.set_ylabel(r'$\theta_0 (intercept)$')
ax.set_zlabel(r'$J(\theta)$')

ax.xaxis.set_major_locator(plt.MaxNLocator(6))
ax.yaxis.set_major_locator(plt.MaxNLocator(6))

ax.view_init(45, 30)

def get_line():
    lineData = np.empty((3, theta_values.shape[0]))
    lineData[0:1,:] = (theta_values[:,0]).T
    lineData[1:2,:] = (theta_values[:,1]).T
    lineData[2:3,:] = loss_values.reshape((theta_values.shape[0],1)).T
    return lineData

def update_lines(num, data, lines):
    for line, dat in zip(lines, data):
        line.set_data(dat[0:2, :num])
        line.set_3d_properties(dat[2, :num])
    return lines

data = [get_line()]
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], 'k-', label = 'Gradient Descent', lw = 1.5)[0] for dat in data]

anim1 = animation.FuncAnimation(fig, update_lines, frames=theta_values.shape[0], fargs=(data, lines), interval=20, repeat=False, blit=True)
plt.show()
# wv = animation.FFMpegWriter(fps=60)
# anim1.save('anim1.mp4', writer=wv)


# CONTOUR ANIMATION
fig = plt.figure(5)
ax = plt.gca()
plt.xlabel(r'$\theta_1 (slope)$')
plt.ylabel(r'$\theta_0 (intercept)$')

x1_range = np.linspace(-1, 1, 100)
x2_range = np.linspace(0, 2, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = J_values(X1, X2)
plt.contour(X1, X2, Z, levels=25, cmap="RdGy")

plt.title("J vs. Theta Contours")

x1 = np.linspace(-1, 1, 30)
x2 = np.linspace(0, 2, 30)

X1, X2 = np.meshgrid(x1, x2)
Z = J_values(X1, X2)

line, = ax.plot([], [], 'k', label = 'Gradient Descent', lw = 1.5)
point, = ax.plot([], [], '*', color = 'b', markersize = 4)
value_display = ax.text(0.02, 0.02, '', transform=ax.transAxes)

def init_2():
    line.set_data([], [])
    point.set_data([], [])
    value_display.set_text('')
    return line, point, value_display

def animate_2(i):
    line.set_data(theta_values[:i,0], theta_values[:i,1])
    point.set_data(theta_values[i,0], theta_values[i,1])
    value_display.set_text("Cost = " + str(loss_values[i]))
    return line, point, value_display

ax.legend(loc = 1)

anim2 = animation.FuncAnimation(fig, animate_2, init_func=init_2, frames=theta_values.shape[0], interval=50, repeat=False, blit=True)
plt.show()
# wv = animation.FFMpegWriter(fps=80)
# anim2.save('anim2.mp4', writer=wv)









