#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
from matplotlib import animation

# In[2]:


m = 1000000 # no. of examples
n = 2 # no. of dimensions of a feature (excluding x0)
theta = np.array([[3], [1], [2]], dtype=np.float64)


# In[3]:


np.random.seed(42)


x1 = np.random.normal(loc=3, scale=2, size=(m,1))
x2 = np.random.normal(loc=-1, scale=2, size=(m,1))
# print(x1.shape)
# print(x2.shape)
# print(x1)
# print(x2)



# In[4]:


# generating the design matrix X
X = np.concatenate((x1, x2), axis=1)
X = np.concatenate((np.ones(shape=(m, 1), dtype=np.float64), X), axis=1)
# print(X.shape, X)


# In[5]:


# generating sampled data Y with Y(i) = N(X*theta, variance=2)
noise_variance = 2
Y_means = np.matmul(X, theta)
# print(Y_means.shape, Y_means)
Y = np.random.normal(loc=Y_means, scale=np.sqrt(noise_variance), size=(m,1))
# print(Y.shape)
# print(Y)


# In[6]:


# shuffling data
perm = np.random.permutation(m)
Y = Y[perm]
X = X[perm]


# In[7]:


# define loss function J(theta) 
def J(theta, X, Y, batch_size, batch_number):
    # shapes:    theta: (n+1)*1
    #            X[s:e, :]: batch_size*(n+1)          
    #            Y[s:e, :]: batch_size*1 
    start_ind = batch_size*batch_number
    end_ind = start_ind+batch_size
    difference = Y[start_ind:end_ind,:] - np.matmul(X[start_ind:end_ind,:], theta)
    loss = (1/(2*batch_size)) * np.matmul(difference.T, difference)
    return loss.item()  


# In[8]:


# derivative(gradient) of loss J(theta) wrt. theta at given theta 
def derJ(theta, X, Y, batch_size, batch_number):
    # shapes:    X[s:e, :].T: (n+1)*batch_size          
    #            Y[s:e, :]: batch_size*1
    #            theta: (n+1)*1
    #            np.matmul(X[s:e, :], theta): m*1 
    #            der: (n+1)*1, same as theta 
    start_ind = batch_size*batch_number
    end_ind = start_ind+batch_size
    der = (-1/batch_size) * np.matmul(X[start_ind:end_ind,:].T, Y[start_ind:end_ind,:] - np.matmul(X[start_ind:end_ind,:], theta))
    return der


# In[9]:


# define convergence criteria 
def converged(next_avg_loss, avg_loss, error_threshold):
    loss_change = abs(next_avg_loss - avg_loss)
    # print("\tAbsolute Change in Average Loss:", loss_change, '\n')
    if loss_change < error_threshold:
        return True
    return False  


# In[10]:


batch_sizes = np.array([1, 100, 10000, 1000000])
delta_array = np.array([1e-3, 1e-3, 1e-4, 1e-4]) # loss change threshold for convergence
num_iters = np.array([1000, 1000, 100, 10]) # we check for convergence from average loss of previous num_iter iterations
# learning_rates = np.array([0.001, 0.001, 1e-5,1e-7])
learning_rate = 0.001
learned_theta_dict = {}
loss_v_dict = {}
theta_v_dict = {}
time_dict = {}
iterations_dict = {}

for i in range(len(batch_sizes)):

    batch_size = batch_sizes[i]
    
    print("\nBatch Size:", batch_size, "\n")
    b = m//batch_size

    # eta = learning_rates[i] # learning rate
    eta = learning_rate
    delta = delta_array[i]
    num_iter = num_iters[i]
    
    loss_values = np.empty(shape=[0,], dtype=np.float64)
    theta_values = np.empty(shape=[0,(n+1),1], dtype=np.float64)
    
    learned_theta = np.zeros(shape=((n+1),1), dtype=np.float64)

    loss = J(learned_theta, X, Y, m, 0)
    print("\tInitially, Total Loss =", loss)
    
    # SGD
    start_time = time.time()    
    t = 1 # iteration number    
    has_converged = False # flag for convergence
    
    while True:
        
        for batch_number in range(b):
            
            next_theta = learned_theta - eta * derJ(learned_theta, X, Y, batch_size, batch_number) 

            loss = J(next_theta, X, Y, batch_size, batch_number)
            if t%100==0:
                print("\tIteration", t, ": Loss =", loss)
        
            loss_v = np.reshape(loss, newshape=(1,))
            theta_v = np.reshape(next_theta, newshape=(1,(n+1),1))
            loss_values = np.append(loss_values, loss_v, axis=0)
            theta_values = np.append(theta_values, theta_v, axis=0)

            learned_theta = next_theta
        
            if t%num_iter == 0 and t >= 2*num_iter:
                avg_loss = np.mean(loss_values[t-2*(num_iter):t-num_iter])
                next_avg_loss = np.mean(loss_values[t-num_iter:t])
                if converged(next_avg_loss, avg_loss, delta):
                    has_converged = True 
                    break
            
            t = t+1
            
        if has_converged:
            break 

    total_time = time.time() - start_time           
    
    print("\n\tConverged in", t, "Iterations\n")   
    print("\tTime Taken =", total_time, "seconds\n")
    print("\tFinal Prediction Loss =", J(theta, X, Y, m, 0), '\n')    
    print("\tOptimized Theta:\n", learned_theta, '\n')
    learned_theta_dict[batch_size] = learned_theta 
    loss_v_dict[batch_size] = loss_values
    theta_v_dict[batch_size] = theta_values
    iterations_dict[batch_size] = t
    time_dict[batch_size] = total_time
    
print("\nTheta Learned for Different Batch Sizes:\n", learned_theta_dict)
print("\nTime Taken for Different Batch Sizes:\n", time_dict,)
print("\nIterations Taken for Different Batch Sizes:\n", iterations_dict, '\n')


# In[11]:

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(loss_v_dict[batch_sizes[0]])
axs[0, 0].set_title('Batch Size: '+str(batch_sizes[0]))
axs[0, 1].plot(loss_v_dict[batch_sizes[1]], 'tab:orange')
axs[0, 1].set_title('Batch Size: '+str(batch_sizes[1]))
axs[1, 0].plot(loss_v_dict[batch_sizes[2]],  'tab:green')
axs[1, 0].set_title('Batch Size: '+str(batch_sizes[2]))
axs[1, 1].plot(loss_v_dict[batch_sizes[3]], 'tab:red')
axs[1, 1].set_title('Batch Size: '+str(batch_sizes[3]))

for ax in axs.flat:
    ax.set(xlabel='Iteration', ylabel='Loss')

plt.gcf().set_size_inches(8, 7)
plt.savefig('2.png', dpi=100)
plt.show()


# In[12]:


# Test Error
test_dataframe = pd.read_csv('../data/q2/q2test.csv', header=0, names=['x1', 'x2', 'y'], dtype={'X':np.float64})
x1_test = test_dataframe['x1'].to_numpy()
x2_test = test_dataframe['x2'].to_numpy()
Y_test = test_dataframe['y'].to_numpy()

x1_test = np.reshape(x1_test, newshape=(x1_test.shape[0], 1))
x2_test = np.reshape(x2_test, newshape=(x2_test.shape[0], 1))
Y_test = np.reshape(Y_test, newshape=(Y_test.shape[0], 1))

X_test = np.concatenate((x1_test, x2_test), axis=1)
X_test = np.concatenate((np.ones(shape=(X_test.shape[0], 1), dtype=np.float64), X_test), axis=1)

print("Test Error computed using:")
print("Original Hypothesis:", J(theta, X_test, Y_test, Y_test.size, 0))
print("Batch Size 1 Hypothesis:", J(learned_theta_dict[1], X_test, Y_test, Y_test.size, 0))
print("Batch Size 100 Hypothesis:", J(learned_theta_dict[100], X_test, Y_test, Y_test.size, 0))
print("Batch Size 10000 Hypothesis:", J(learned_theta_dict[10000], X_test, Y_test, Y_test.size, 0))
print("Batch Size 1000000 Hypothesis:", J(learned_theta_dict[1000000], X_test, Y_test, Y_test.size, 0))


# In[13]:


# 3D ANIMATIONS
fig = plt.figure(1)
ax = plt.axes(projection='3d')
ax.autoscale(enable=True, axis='both', tight=True)

ax.set_xlim3d([-0.5, 2.0])
ax.set_ylim3d([-0.5, 4.0])
ax.set_zlim3d([-0.5, 6.0])

ax.set_xlabel(r'$\theta_1 $')
ax.set_ylabel(r'$\theta_2 $')
ax.set_zlabel(r'$\theta_0 $')
ax.set_title('SGD for Batch Size 1')

ax.xaxis.set_major_locator(plt.MaxNLocator(6))
ax.yaxis.set_major_locator(plt.MaxNLocator(6))

ax.view_init(10, -30)

def get_line():
    tv = theta_v_dict[1]
    lineData = np.empty((3, tv.shape[0]))
    lineData[0:1,:] = (tv[:,1]).T
    lineData[1:2,:] = (tv[:,2]).T
    lineData[2:3,:] = (tv[:,0]).T
    return lineData

def update_lines(num, data, lines):
    for line, dat in zip(lines, data):
        line.set_data(dat[0:2, :num])
        line.set_3d_properties(dat[2, :num])
    return lines

data = [get_line()]
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], 'b-', label = 'Gradient Descent', lw = 1.5)[0] for dat in data]

anim1 = animation.FuncAnimation(fig, update_lines, frames=theta_values.shape[0], fargs=(data, lines), interval=0.5, repeat=False, blit=True)
plt.show()
# wv = animation.FFMpegWriter(fps=60)
# anim1.save('anim1.mp4', writer=wv)



fig = plt.figure(2)
ax = plt.axes(projection='3d')
ax.autoscale(enable=True, axis='both', tight=True)

ax.set_xlim3d([-0.5, 2.0])
ax.set_ylim3d([-0.5, 4.0])
ax.set_zlim3d([-0.5, 6.0])

ax.set_xlabel(r'$\theta_1 $')
ax.set_ylabel(r'$\theta_2 $')
ax.set_zlabel(r'$\theta_0 $')
ax.set_title('SGD for Batch Size 100')

ax.xaxis.set_major_locator(plt.MaxNLocator(6))
ax.yaxis.set_major_locator(plt.MaxNLocator(6))

ax.view_init(10, -30)

def get_line():
    tv = theta_v_dict[100]
    lineData = np.empty((3, tv.shape[0]))
    lineData[0:1,:] = (tv[:,1]).T
    lineData[1:2,:] = (tv[:,2]).T
    lineData[2:3,:] = (tv[:,0]).T
    return lineData

def update_lines(num, data, lines):
    for line, dat in zip(lines, data):
        line.set_data(dat[0:2, :num])
        line.set_3d_properties(dat[2, :num])
    return lines

data = [get_line()]
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], 'b-', label = 'Gradient Descent', lw = 1.5)[0] for dat in data]

anim1 = animation.FuncAnimation(fig, update_lines, frames=theta_values.shape[0], fargs=(data, lines), interval=0.5, repeat=False, blit=True)
plt.show()
# wv = animation.FFMpegWriter(fps=60)
# anim1.save('anim2.mp4', writer=wv)



fig = plt.figure(3)
ax = plt.axes(projection='3d')
ax.autoscale(enable=True, axis='both', tight=True)

ax.set_xlim3d([-0.5, 2.0])
ax.set_ylim3d([-0.5, 4.0])
ax.set_zlim3d([-0.5, 6.0])

ax.set_xlabel(r'$\theta_1 $')
ax.set_ylabel(r'$\theta_2 $')
ax.set_zlabel(r'$\theta_0 $')
ax.set_title('SGD for Batch Size 10000')

ax.xaxis.set_major_locator(plt.MaxNLocator(6))
ax.yaxis.set_major_locator(plt.MaxNLocator(6))

ax.view_init(10, -30)

def get_line():
    tv = theta_v_dict[10000]
    lineData = np.empty((3, tv.shape[0]))
    lineData[0:1,:] = (tv[:,1]).T
    lineData[1:2,:] = (tv[:,2]).T
    lineData[2:3,:] = (tv[:,0]).T
    return lineData

def update_lines(num, data, lines):
    for line, dat in zip(lines, data):
        line.set_data(dat[0:2, :num])
        line.set_3d_properties(dat[2, :num])
    return lines

data = [get_line()]
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], 'b-', label = 'Gradient Descent', lw = 1.5)[0] for dat in data]

anim1 = animation.FuncAnimation(fig, update_lines, frames=theta_values.shape[0], fargs=(data, lines), interval=0.5, repeat=False, blit=True)
plt.show()
# wv = animation.FFMpegWriter(fps=60)
# anim1.save('anim3.mp4', writer=wv)



fig = plt.figure(4)
ax = plt.axes(projection='3d')
ax.autoscale(enable=True, axis='both', tight=True)

ax.set_xlim3d([-0.5, 2.0])
ax.set_ylim3d([-0.5, 4.0])
ax.set_zlim3d([-0.5, 6.0])

ax.set_xlabel(r'$\theta_1 $')
ax.set_ylabel(r'$\theta_2 $')
ax.set_zlabel(r'$\theta_0 $')
ax.set_title('SGD for Batch Size 1000000')

ax.xaxis.set_major_locator(plt.MaxNLocator(6))
ax.yaxis.set_major_locator(plt.MaxNLocator(6))

ax.view_init(10, -30)

def get_line():
    tv = theta_v_dict[1000000]
    lineData = np.empty((3, tv.shape[0]))
    lineData[0:1,:] = (tv[:,1]).T
    lineData[1:2,:] = (tv[:,2]).T
    lineData[2:3,:] = (tv[:,0]).T
    return lineData

def update_lines(num, data, lines):
    for line, dat in zip(lines, data):
        line.set_data(dat[0:2, :num])
        line.set_3d_properties(dat[2, :num])
    return lines

data = [get_line()]
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], 'b-', label = 'Gradient Descent', lw = 1.5)[0] for dat in data]

anim1 = animation.FuncAnimation(fig, update_lines, frames=theta_values.shape[0], fargs=(data, lines), interval=0.5, repeat=False, blit=True)
plt.show()
# wv = animation.FFMpegWriter(fps=60)
# anim1.save('anim4.mp4', writer=wv)




