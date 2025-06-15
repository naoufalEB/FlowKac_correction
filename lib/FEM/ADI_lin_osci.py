# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time

def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n-1):
        denom = b[i] - a[i] * c_prime[i-1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom
    
    d_prime[n] = (d[n] - a[n] * d_prime[n-1]) / (b[n] - a[n] * c_prime[n-1])
    #for i in range(1, n):
    #    d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / (b[i] - a[i] * c_prime[i-1])

    x = np.zeros(n)
    x[-1] = d_prime[-1]

    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]

    return x

def adi_lin_osci(p0, x1, x2, dx, dt, Tmax):
    n_steps = int(Tmax/dt)
    
    p_t = np.zeros((p0.shape[0], p0.shape[1], n_steps))
    p_t[...,0] = p0
    
    nx = len(x1)
    ny = len(x2)
    h = dt/2
    
    start_time = time.time()
    for n in range(n_steps-1):
        
        b = 1
        p_star = np.zeros_like(p0)
        
        for j in range(1, ny-1):
            a = np.full(nx-2, -x2[j]*h/(2*dx))
            c = -a
            d = p_t[1:-1,j, n]*(1 + 0.2*h) + (0.2*x2[j] + x1[1:-1])*h*(p_t[1:-1,j+1, n] - p_t[1:-1,j-1, n])/(2*dx) + 0.2*h*(p_t[1:-1,j+1 , n] - 2*p_t[1:-1,j, n] + p_t[1:-1,j-1, n])/(dx**2)
            p_star[1:-1, j] = thomas_algorithm(a, np.full(nx-2,b), c, d)
        
        b = (1 + 0.4*h/(dx**2))
        for i in range(1, nx-1):
            a = (0.2*x2[1:-1] + x1[i])*h/(2*dx) - 0.2*h/(dx**2)
            c = -(0.2*x2[1:-1] + x1[i])*h/(2*dx) - 0.2*h/(dx**2)
            d = p_star[i, 1:-1]*(1 + 0.2*h) - x2[1:-1]*(p_star[i + 1, 1:-1] - p_star[i - 1, 1:-1])*h/(2*dx)
            p_t[i, 1:-1 ,n+1] = thomas_algorithm(a, np.full(ny-2,b), c, d)        
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Linear ADI : {:.2f} seconds".format(execution_time))
    
    p_t_transposed = np.transpose(p_t, axes=(1, 0, 2))
    return p_t_transposed

# dt = 0.001
# Tmax = 3

# dx = dy = 0.05
# x1 = np.arange(-3, 3, step = dx)
# x2 = np.arange(-3, 3, step = dx)
# X, Y = np.meshgrid(x1, x2, indexing = "ij")
# pos = np.dstack((X, Y))

# mean = [1, 1]
# cov = np.eye(2)/9

# dist = multivariate_normal(mean, cov)
# p0 = dist.pdf(pos)

# p_solved  = adi_lin_osci(p0, x1, x2, dx, dt, Tmax)

# plt.imshow(p_solved[...,0], extent=[-3, 3, -3, 3], origin='lower', cmap='viridis')
# plt.title('Initial State')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.imshow(p_solved[...,1500].T, extent=[-3, 3, -3, 3], origin='lower', cmap='viridis')
# plt.title('state 1.5')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.imshow(p_solved[...,1500], extent=[-3, 3, -3, 3], origin='lower', cmap='viridis')
# plt.title('state 1.5')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.imshow(p_solved[...,2999], extent=[-3, 3, -3, 3], origin='lower', cmap='viridis')
# plt.title('state 3.')
# plt.xlabel('x')
# plt.ylabel('y')