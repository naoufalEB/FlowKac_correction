# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time
import torch
from numba import jit, prange
import os
import pickle

@jit(nopython=True)
def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n-1):
        denom = (b[i] - a[i] * c_prime[i-1])
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom
        
    # for i in range(1, n):
    #     d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / (b[i] - a[i] * c_prime[i-1])
    d_prime[-1] = (d[-1] - a[-1] * d_prime[-2]) / (b[-1] - a[-1] * c_prime[-2])

    x = np.zeros(n)
    x[-1] = d_prime[-1]

    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]

    return x

@jit(nopython=True, parallel=True)
def adi_nonlin_osci(x1, x2, Tmax, pos):
    
    dt = 0.005
    x1_np = x1#.detach().cpu().numpy()
    x2_np = x2#.detach().cpu().numpy()
    dx = x1_np[1] - x1_np[0]
    nx = len(x1_np)
    ny = len(x2_np)
    
    n_steps = int(Tmax/dt)
    
    
    mean = np.array([0, 8])
    #cov = np.eye(2)
    # X, Y = np.meshgrid(x1_np, x2_np, indexing="ij")
    # pos = np.dstack((X, Y))
    
    
    #p0 = (1/(2*np.pi))*np.exp(-np.sum((pos - mean)**2, axis = -1)/2)
    
    #dist = multivariate_normal(mean, cov)
    #p0 = dist.pdf(pos)
    
    p_t = np.zeros((nx, ny, n_steps))
    
    #adapted_pos = np.transpose(pos, axes = (1, 0, 2))
    
    p_t[...,0] = (1/(np.pi))*np.exp(-np.sum((pos - mean)**2, axis = -1))#p0
    
    
    h = dt/2
    
    x1_cube = x1**3
    
    
    
    for n in range(n_steps-1):
        
        print("ops: " + str(n) + "/" + str(n_steps))
        b = np.full(nx-2,1)
        p_star = np.zeros_like(p_t[...,0])
        
        for j in range(1, ny-1):
            a = np.full(nx-2, -x2_np[j]*h/(2*dx))
            c = -a
            
            dp_t = (p_t[1:-1,j+1, n] - p_t[1:-1,j-1, n])/(2*dx)
            ddp_t = (p_t[1:-1,j+1 , n] - 2*p_t[1:-1,j, n] + p_t[1:-1,j-1, n])/(dx**2)
            d = p_t[1:-1,j, n]*(1 + 0.4*h) - (x1_np[1:-1] - 0.4*x2_np[j] -0.1*x1_cube[1:-1] )*h*dp_t + 0.4*h*ddp_t
            p_star[1:-1, j] = thomas_algorithm(a, b, c, d)
        
        b = np.full(ny-2, 1 + 0.8*h/(dx**2))
        for i in range(1, nx-1):
            a = -(x1_np[i] - 0.4*x2_np[1:-1] - 0.1*x1_cube[i] )*h/(2*dx) - 0.4*h/(dx**2)
            c = (x1_np[i] - 0.4*x2_np[1:-1] - 0.1*x1_cube[i] )*h/(2*dx) - 0.4*h/(dx**2)
            d = p_star[i, 1:-1]*(1 + 0.4*h) - x2_np[1:-1]*(p_star[i + 1, 1:-1] - p_star[i - 1, 1:-1])*h/(2*dx)
            p_t[i, 1:-1 ,n+1] = thomas_algorithm(a, b, c, d)        
    
        
    #p_t_transposed = np.transpose(p_t, axes=(1, 0, 2))
    
    return np.clip(p_t, 0, None)


# Tmax = 3

# #dx = dy = 0.1
# # #x1 = np.arange(-10, 10, step = dx)
# # #x2 = np.arange(-10, 10, step = dx)
# x1 = np.linspace(-10, 10, 150)
# x2 = np.linspace(-10, 10, 150)

# X, Y = np.meshgrid(x1, x2, indexing="ij")
# pos = np.dstack((X, Y))

# start_time = time.time()
# p_solved  = adi_nonlin_osci(x1, x2, Tmax, pos)
# end_time = time.time()

# # index_extraction = np.arange(0, 201, 4) # creer une grille 51x51 a partir de 201x201
# # p_solved_extracted = p_solved[index_extraction][:, index_extraction]

# # density_path = 'C:/Users/naouf/Documents/0-These/2-Scripts/12-Temporal_NF/FC_TNF/fem_results/nonlinear_osci'
# # pickle_file_path = os.path.join(density_path, 'density_estimation.pkl')

# # with open(pickle_file_path, 'wb') as file:
# #     pickle.dump(p_solved_extracted, file)


# execution_time = end_time - start_time
# print("non Linear ADI : {:.2f} seconds".format(execution_time))

# plt.figure()
# plt.imshow(p_solved[...,0], extent=[-10, 10, -10, 10], origin='lower', cmap='Reds')
# plt.title('Initial State')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.figure()
# plt.imshow(p_solved[...,199], extent=[-10, 10, -10, 10], origin='lower', cmap='Reds')
# plt.title('state 1 trans')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.figure()
# plt.imshow(p_solved[...,399], extent=[-10, 10, -10, 10], origin='lower', cmap='Reds')
# plt.title('state 1 trans')
# plt.xlabel('x')
# plt.ylabel('y')


# plt.figure()
# plt.imshow(p_solved[...,599], extent=[-10, 10, -10, 10], origin='lower', cmap='Reds')
# plt.title('state 1 trans')
# plt.xlabel('x')
# plt.ylabel('y')


# p_array_test = p_solved[...,199]/np.sum(p_solved[...,199])
# flat = p_array_test.flatten()

# sample_index = np.random.choice(a=flat.size, p=flat, size = 1000)

# # Take this index and adjust it so it matches the original array
# adjusted_index = np.unravel_index(sample_index, p_array_test.shape)
# adjusted_index = np.array(list(zip(*adjusted_index)))
# test = pos[adjusted_index[:,0], adjusted_index[:,1],:]
# x_test = test[:,0]
# y_test = test[:,1]

# plt.figure(figsize=(8, 6))
# plt.scatter(x_test, y_test, c='blue', alpha=0.5)
# plt.title('Scatter Plot of 2D Points')
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid(True)
# plt.show()

# plt.figure()
# plt.imshow(p_solved[...,399], extent=[-10, 10, -10, 10], origin='lower', cmap='Reds')
# plt.title('state 2')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.figure()
# plt.imshow(p_solved[...,599], extent=[-10, 10, -10, 10], origin='lower', cmap='Reds')
# plt.title('state 3.')
# plt.xlabel('x')
# plt.ylabel('y')



# i = 399
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, p_solved[...,i], cmap='Reds')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('Density')
# ax.set_title('state 1')
