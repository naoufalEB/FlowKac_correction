# -*- coding: utf-8 -*-



import torch
import math
from torch import nn
import numpy as np
from scipy.linalg import expm
from scipy.stats import multivariate_normal
import time
import lib.FEM as FEM


def init_sde(args):
    sde_classes = {
        "gbm_1d": SDE_gbm_1d,
        "gbm_2d": SDE_gbm_2d,
        "gbm_nd": SDE_gbm_nd,
        "lin_osci": SDE_linear_osci,
        "nonlin_osci": SDE_nonlinear_osci,
        "ou_nd": SDE_OU_nd
    }
    
    try:
        return sde_classes[args.sde_name](args)
    except KeyError:
        raise NotImplementedError(f"SDE model '{args.sde_name}' is not implemented.")
        
class SDE_gbm_1d(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sde_type = "ito"
        self.noise_type = "diagonal"
        self.mu = 0.5
        self.sigma = 1
        self.device = torch.device(args.device)

    def u(self, t, y): # SDE true drift
        return torch.ones_like(y)*y*self.mu
    
    def f(self, t, y): # FlowKac drift
        return -self.u(t,y) + 2*self.g(t,y)*self.sigma
    
    def g(self, t, y): # SDE vol
        return torch.ones_like(y)*y*self.sigma
    
    def v(self, t, y): # q function
        return torch.ones_like(y)*(self.mu - self.sigma**2)
    
    def density(self, t, y, args):
        log_mean = (self.mu - 0.5*(self.sigma**2))*t
        log_sd = torch.sqrt(t+1)*self.sigma
        
        return (1/(math.sqrt(2*math.pi)*y*log_sd)) * torch.exp(-torch.sum((torch.log(y) - log_mean)**2, dim=-1, keepdim=True)/(2*(log_sd**2)))

class SDE_gbm_2d(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sde_type = "ito"
        self.noise_type = "scalar"
        
        self.device = torch.device(args.device)
        
        self.B = torch.tensor([[0.5, 0.],[0., 1.]], device = self.device)
        self.A = torch.tensor([[-1., 0],[0., -2.]], device = self.device)
        
        self.Mu = torch.tensor([0.5, 0.7], device = self.device)
        self.Sigma = 0.5*torch.eye(2, device = self.device)
        
        self.drift_matrix = self.A + 0.5 * torch.matmul(self.B, self.B)
        self.q = torch.trace(self.A + self.B/2) - torch.trace(self.B**2) - self.B[0,0]*self.B[1,1] - self.B[0,1]*self.B[1,0]
    
    def u(self, t, y): # SDE true drift
        
        drift_matrix_expanded = self.drift_matrix.unsqueeze(0).expand(y.shape[0], -1, -1)
                
        return torch.matmul(drift_matrix_expanded, y.unsqueeze(-1)).squeeze(-1)
    
    def f(self, t, y): # FlowKAc drift
        derivativeD_xj = torch.zeros_like(y, device = self.device)
        y1, y2 = y.unbind(dim=1)
        b11 = self.B[0,0]
        b12 = self.B[0,1]
        b22 = self.B[1,1]
        
        derivativeD_xj[:,0] = 2*b11*(b11*y[:,0] + b12*y[:,1]) + (b11*b22 + b12*b12)*y[:,0] + 2*b12*b22*y[:,1]
        derivativeD_xj[:,1] = 2*b22*(b12*y[:,0] + b22*y[:,1]) + (b22*b11 + b12*b12)*y[:,1] + 2*b12*b11*y[:,0]
        
        return -self.u(t,y) + derivativeD_xj
    
    def g(self, t, y): # SDE volatility
        return torch.matmul(self.B.unsqueeze(0).expand(y.shape[0], -1, -1), y.unsqueeze(-1))
    
    def v(self, t, y): # q function
        
        return self.q*torch.ones_like(torch.sum(y, dim = -1, keepdim = True))
    
    @torch.no_grad()
    def density(self, t, y, args):
        
        
        if t==0:
            return 1/(math.pi) * torch.exp(-torch.sum((torch.log(y) - self.Mu)**2, dim=-1, keepdim=True))/torch.prod(y, dim = -1, keepdim=True)
        else:
            
            log_Mu = self.Mu + torch.diag(self.A)*t
            log_Sigma = self.Sigma + (self.B**2)*t
            log_Sigma[0,1] = torch.prod(torch.diag(self.B))*t
            log_Sigma[1,0] = torch.prod(torch.diag(self.B))*t
            det_Sigma = torch.det(log_Sigma)
            inv_Sigma = torch.inverse(log_Sigma)
            
            exponent = -0.5 * torch.sum((torch.log(y) - log_Mu) @ inv_Sigma * (torch.log(y)- log_Mu), dim=-1, keepdim=True)
            p = 1/(2*math.pi*torch.sqrt(det_Sigma)) * torch.exp(exponent)/torch.prod(y, dim = -1, keepdim=True)
            return p
        

class SDE_gbm_nd(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sde_type = "ito"
        self.noise_type = "scalar"
        
        self.device = torch.device(args.device)
        self.d = args.Dx
        self.a = -1.0
        self.b = 0.5
        self.sigma0 = 0.5
        mu0 = 0.7
        
        
        self.B = self.b*torch.eye(args.Dx, device = self.device)
        self.A = self.a*torch.eye(args.Dx, device = self.device)
        
        self.Mu = mu0*torch.ones(args.Dx, device = self.device)
        self.Sigma = self.sigma0*torch.eye(args.Dx, device = self.device)
        
        self.drift_matrix = self.A + 0.5 * torch.matmul(self.B, self.B)
    
    def u(self, t, y): # SDE drift
        
        drift_matrix_expanded = self.drift_matrix.unsqueeze(0).expand(y.shape[0], -1, -1)       
        return torch.matmul(drift_matrix_expanded, y.unsqueeze(-1)).squeeze(-1)
    
    def f(self, t, y): # FlowKac drift
        
        return -self.u(t,y) + (self.b**2)*(self.d + 1)*y
    
    def g(self, t, y): # SDE volatility
        return torch.matmul(self.B.unsqueeze(0).expand(y.shape[0], -1, -1), y.unsqueeze(-1))
    
    def v(self, t, y): # q function
        return (self.a*self.d - 0.5*(self.d*self.b)**2)*torch.ones_like(torch.sum(y, dim = -1, keepdim = True))
    
    @torch.no_grad()
    def density(self, t, y, args):
        
        if t==0:
            det_Sigma = torch.det(self.Sigma)
            return ((2*math.pi)**(-self.d/2))* (det_Sigma**(-0.5)) * torch.exp(-0.5*(self.sigma0**(-1)) * torch.sum((torch.log(y) - self.Mu)**2, dim=-1, keepdim=True))/torch.prod(y, dim = -1, keepdim=True)
        else:
            
            log_Mu = self.Mu + self.a*t
            log_Sigma = self.Sigma + (self.b**2)*t
            det_Sigma = torch.det(log_Sigma)
            inv_Sigma = torch.inverse(log_Sigma)
            
            exponent = -0.5 * torch.sum((torch.log(y) - log_Mu) @ inv_Sigma * (torch.log(y)- log_Mu), dim=-1, keepdim=True)
            p = ((2*math.pi)**(-self.d/2))* (det_Sigma**(-0.5)) * torch.exp(exponent)/torch.prod(y, dim = -1, keepdim=True)
            return p
    
    def sample_test_data(self, t, num_samples):
        
        log_Mu = self.Mu + self.a*t
        log_Sigma = self.Sigma + (self.b**2)*t
        mvn = torch.distributions.MultivariateNormal(log_Mu, log_Sigma)
        normal_samples = mvn.sample((num_samples,))
        log_normal_samples = torch.exp(normal_samples)
        
        return log_normal_samples
    

class SDE_linear_osci(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sde_type = "ito"
        self.noise_type = "diagonal"
        self.sigma = torch.sqrt(torch.tensor([0.6, 0.]))
                
        self.theta = torch.tensor([[0.1, 1.],[-1., -0.1]])
        
        self.m_0 = np.array([1., 1.])
        self.V_0 = np.eye(2)/9
        
    def u(self, t, y): # SDE drift
        
        theta = self.theta.unsqueeze(0).expand(y.size(0), -1, -1).to(y)
        return torch.matmul(theta, y.unsqueeze(-1)).squeeze(-1)
        
    def f(self, t, y): # FlowKac drift
        return -self.u(t,y)
    
    def g(self, t, y):  # SDE Diffusion/volatility
        return torch.ones_like(y)*self.sigma.to(y)
    
    def v(self, t, y): # q function
        return (self.theta[0,0] + self.theta[1,1])*torch.ones_like(torch.sum(y, dim = -1, keepdim = True))
        
    def approximate_density(self, t_element, Dx_grid):
        
        theta = self.theta.detach().cpu().numpy()
        Sigma = torch.diag(self.sigma).detach().cpu().numpy()
                
        m_0 = self.m_0
        V_0 = self.V_0
        
        t_integration = np.linspace(0, t_element.detach().cpu().numpy(), 101)
        
        Sig_SigT = np.repeat(np.matmul(Sigma, Sigma.T)[None,:,:], len(t_integration), axis = 0)
        exp_negtheta_r = expm(-t_integration[:,None,None]*theta[None,:,:])
        exp_negtheta_T_r = expm(-t_integration[:,None,None]*theta.T[None,:,:])
        
        exp_theta_t = expm(t_integration[-1]*theta)
        exp_theta_T_t = expm(t_integration[-1]*theta.T)
        
        integrand = np.matmul(np.matmul(exp_negtheta_r, Sig_SigT), exp_negtheta_T_r)
        
        V_t = np.matmul(np.matmul(exp_theta_t, V_0 + np.trapz(integrand, t_integration, axis = 0)), exp_theta_T_t)
        m_t = np.matmul(exp_theta_t, m_0)
        
        
        V_t_sym = (V_t + V_t.T)/2 # force symmetry of V    
        
        
        normal_distributions = multivariate_normal(mean=m_t, cov=V_t_sym)
        pdf_values_grid = normal_distributions.pdf(Dx_grid)
        
        return pdf_values_grid
        
    def density(self, t, y, args):
        
        mu = torch.as_tensor(self.m_0).to(y)
        p0 = (9/(2*math.pi))*torch.exp(-9*torch.sum((y - mu)**2, dim = -1, keepdim = True)/2)
        
        if t == 0:
            return p0
        else:                      
            reshaped_data = y.view(args.x_num_test, args.x_num_test, args.Dx)
            
            p = self.approximate_density(t, reshaped_data.detach().cpu().numpy())
            p = torch.as_tensor(p, device = t.device).view(-1, 1)
                        
            return p

class SDE_nonlinear_osci(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sde_type = "ito"
        self.noise_type = "diagonal"
        self.sigma = torch.sqrt(torch.tensor([0., 0.8]))
                
        self.T_max = args.T_max
        self.estimated_density = None
        self.initialized = False
        self.adi_t_vec = None
                
        self.initialized = False

    def u(self, t, y): # SDE drift
        y1, y2 = torch.split(y, split_size_or_sections=(1, 1), dim=1)
        u1 = y2
        u2 = y1 - 0.4*y2 - 0.1*(y1**3)
        return torch.cat([u1, u2], dim=1)
        
    def f(self, t, y): # FlowKac drift : -mu  + 2 d/dx_j (D_ij)
        return -self.u(t,y)
    
    def g(self, t, y):  # Diffusion function
        return torch.ones_like(y)*self.sigma.to(y)
    
    def v(self, t, y): # q function : d/dx(mu) - d^2/dx_idx_j (D_ij)
        return torch.sum(torch.ones_like(y)*torch.tensor([0., -0.4], device = t.device), dim = -1, keepdim = True)
        
    
    def approximate_density(self, x1, x2):
        if not self.initialized:
            
            x1_np = x1.detach().cpu().numpy()
            x2_np = x2.detach().cpu().numpy()
            
            X1_np, X2_np = np.meshgrid(x1_np, x2_np, indexing = "ij")
            temp_pos = np.dstack((X1_np, X2_np))
            
            start_time = time.time()
            self.estimated_density = FEM.adi_nonlin_osci(x1_np, x2_np, self.T_max, temp_pos)
            self.initialized = True
            end_time = time.time()
            execution_time = end_time - start_time
            print("non Linear ADI : {:.2f} seconds".format(execution_time))
                    
        self.adi_t_vec = np.arange(0, self.T_max, step = 0.005)
        
        return self.estimated_density
    
    def density(self, t, y, args):
        
        mu = torch.tensor([0.,8.]).to(y)
        p0 = (1/(math.pi))*torch.exp(-torch.sum((y - mu)**2, dim = -1, keepdim = True))
        
        if t == 0:
            return p0
        else:
                  
            reshaped_data = y.view(args.x_num_test, args.x_num_test, args.Dx)
            x = reshaped_data[:, 0, 0]
            y = reshaped_data[0, :, 1]
                        
            p = self.approximate_density(x, y)
            closest_index = np.argmin(np.abs(t.detach().cpu().numpy() - self.adi_t_vec))
            p_t = p[..., closest_index]
            p_t = torch.as_tensor(p_t, device = t.device).reshape(-1, 1)
            
            return p_t
        

class SDE_OU_nd(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sde_type = "ito"
        self.noise_type = "diagonal"
        self.sigma = 0.6
        self.a = 0.9
        self.d = args.Dx        
        
        self.device = torch.device(args.device)
        
        self.m_0 = torch.ones(self.d, device = self.device)
        self.V_0 = (1/4)*torch.eye(self.d, device = self.device)
        
    def u(self, t, y): # vrai drift de l'EDS
        return self.a* y
        
    def f(self, t, y): # drift de l'EDS de Feynman-Kac
        return -self.u(t,y)
    
    def g(self, t, y):  # Diffusion function
        return torch.ones_like(y)*self.sigma
    
    def v(self, t, y): # derivative of the drift mu
        return (self.d * self.a)*torch.ones_like(torch.sum(y, dim = -1, keepdim = True))
                
    def density(self, t, y, args):
        
        if t == 0:
            
            p0 = ((2*math.pi/4)**(-self.d/2)) * torch.exp(-4*0.5*torch.sum((y - self.m_0)**2, dim = -1, keepdim = True))
            return p0
        else:
            
            exp_2a_t = torch.exp(2*self.a*t)
            v_t = exp_2a_t/4 + (0.5*(self.sigma**2)/self.a) * (exp_2a_t - 1)
            
            m_t = torch.exp(self.a*t)*self.m_0
            
            p = ((2 * math.pi * v_t)**(-self.d/2)) * torch.exp(-0.5 * (1/v_t) * torch.sum((y - m_t)**2, dim = -1, keepdim = True))

            return p
        
    def sample_test_data(self, t, num_samples):
                
        exp_2a_t = torch.exp(2*self.a*t)
        Id = torch.eye(self.d).to(t)
        
        V_t = exp_2a_t*self.V_0 + (0.5*(self.sigma**2)/self.a) * (exp_2a_t - 1)*Id
        m_t = torch.exp(self.a*t)*self.m_0

        mvn = torch.distributions.MultivariateNormal(m_t, V_t)
        normal_samples = mvn.sample((num_samples,))
        
        return normal_samples