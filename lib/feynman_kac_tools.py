# -*- coding: utf-8 -*-

import torch
import torchsde



def t_sde_sample(t_vec, args):
    """
    Create a finer time vector by subdividing the given time vector based on a multiplier.

    Args:
        t_vec (Tensor): Original time vector.
        args: Configuration object with multiplier_T_num, T_num, and T_max.

    Returns:
        Tuple[Tensor, Tensor]: Subdivided time vector and indices of the original time points.
    """
    new_len = args.multiplier_T_num*(args.T_num - 1) + 1
    new_t_vec = torch.linspace(0, args.T_max, new_len).to(t_vec)
    
    index_init = torch.arange(0, new_len, args.multiplier_T_num)
        
    return new_t_vec, index_init
    

class BaseSampler:
    def __init__(self, sde, t_vec, args):
        self.sde = sde
        self.t_vec = t_vec
        self.args = args
        self.Dx = args.Dx
        self.W_Dx = args.W_Dx
        self.N_sample = args.N_sample
        
        self.args = args
        self.device = torch.device(args.device)
        self.method = "euler"
        self.eps = 1e-5
        
        self.t_vec_subdivised, self.init_index = self._subdivide_time(t_vec)     
                
    def _subdivide_time(self, t_vec):
        new_len = self.args.multiplier_T_num * (self.args.T_num - 1) + 1
        new_t = torch.linspace(0, self.args.T_max, new_len, device=self.device)
        index = torch.arange(0, new_len, self.args.multiplier_T_num, device=self.device)
        return new_t, index
    
    def _expand_x0(self, x0):
        if x0 is None:
            return torch.zeros(self.N_sample, self.Dx, device=self.device)
        return x0.view(x0.shape[0], 1, self.Dx).expand(-1, self.N_sample, -1).reshape(-1, self.Dx)
    
    def _make_bm(self):
        return torchsde.BrownianInterval(
            t0=self.t_vec_subdivised[0],
            t1=self.t_vec_subdivised[-1],
            size=(self.N_sample, self.W_Dx),
            device=self.device
        )
    
    def _reshape_output(self, xhat):
        xhat = xhat[self.init_index]
        xhat = xhat.view(len(self.t_vec), -1, self.N_sample, self.Dx)
        return torch.transpose(xhat, 1, 0)

    def sample(self):
        raise NotImplementedError("Subclasses should implement this method")


class ExactSampler(BaseSampler):
    
    @torch.no_grad()
    def sample(self, x0):
          
        x0 = x0.view(x0.shape[0], 1, self.Dx).expand(-1, self.N_sample, -1).reshape(-1, self.Dx)
        xhat = torchsde.sdeint(self.sde, x0, self.t_vec_subdivised, method=self.method)
        
        xhat = xhat[self.init_index, ...]
        xhat = xhat.view(len(self.t_vec), -1, self.N_sample, self.Dx)
        xhat = torch.transpose(xhat, 1, 0)

        return xhat, None, None

        
class JacobianSampler(BaseSampler):
    """
    Jacobian approximation using finite differences
    """
    @torch.no_grad()
    def sample(self, x0 = None):
        
            
        x0 = self._expand_x0(x0)
        bm = self._make_bm()
        
        xhat = torchsde.sdeint(self.sde, x0, self.t_vec_subdivised, method=self.method, bm = bm)
        jacobian_xhat = torch.zeros((len(self.t_vec_subdivised), xhat.shape[1], self.Dx, self.Dx), device = self.device)#.to(self.t_vec)

        for i in range(self.Dx):
            torch_eps = torch.zeros_like(x0)
            torch_eps[:, i] = self.eps
            xhat_eps = torchsde.sdeint(self.sde, x0 + torch_eps, self.t_vec_subdivised, method=self.method, bm = bm)
            jacobian_xhat[..., i] = (xhat_eps - xhat) / self.eps

        jacobian_xhat = jacobian_xhat[self.init_index, ...]
        
        xhat = xhat[self.init_index, ...]
        xhat = xhat.view(len(self.t_vec), -1, self.N_sample, self.Dx)
        xhat = torch.transpose(xhat, 1, 0)

        return xhat, jacobian_xhat, None

class JacobianSamplerExact(BaseSampler):
    """
    Jacobian computation using autodiff
    """
    @torch.no_grad()
    def sample(self, x0 = None):
        x0 = self._expand_x0(x0)
        bm = self._make_bm()
        
        def compute_x_paths(x0):
            x0_bis = x0.expand(self.N_sample, self.Dx)
            sampled_SDE = torchsde.sdeint(self.sde, x0_bis, self.t_vec_subdivised, bm=bm, method=self.method)
            return sampled_SDE[self.init_index, ...]
        

        jacobian_xhat = torch.func.jacrev(compute_x_paths, argnums=0)(x0)
        
        xhat = xhat = self._reshape_output(
            torchsde.sdeint(self.sde, x0.expand(self.N_sample, self.Dx), self.t_vec_subdivised, method=self.method, bm = bm)
        )
         
        return xhat, jacobian_xhat, None
    
class HessianSamplerExact(BaseSampler):
    """
    Hessian computation using autodiff
    """
    @torch.no_grad()
    def sample(self, x0 = None):
        x0 = self._expand_x0(x0)
        bm = self._make_bm()
        
        def compute_x_paths(x0):
            x0_bis = x0.expand(self.N_sample, self.Dx)
            sampled_SDE = torchsde.sdeint(self.sde, x0_bis, self.t_vec_subdivised, bm=bm, method=self.method)
            return sampled_SDE[self.init_index, ...]
        
        jacobian_xhat = torch.func.jacrev(compute_x_paths, argnums=0)(x0)
        
        # We define a jacobian function, that we differentiate to avoid the Hessian's jacfwd that trigger a randomness error
        def jacobian_fn(x):
            return torch.func.jacrev(compute_x_paths, argnums=0)(x)
        
        hessian_xhat = torch.func.jacrev(jacobian_fn, argnums=0)(x0)
        
        xhat = torchsde.sdeint(self.sde, x0.expand(self.N_sample, self.Dx), self.t_vec_subdivised, method=self.method, bm = bm)
        xhat = xhat[self.init_index, ...]
        xhat = xhat.view(len(self.t_vec), -1, self.N_sample, self.Dx)
        xhat = torch.transpose(xhat, 1, 0)
        
        return xhat, jacobian_xhat, hessian_xhat
    
class HessianSampler(BaseSampler):
    """
    Hessian approximation using finite differences
    """
    def sample(self, x0 = None):
        
        with torch.no_grad():
            
            if x0 is None:
                x0 = torch.zeros(self.N_sample, self.Dx, device = self.device)
            else:
                x0 = x0.view(x0.shape[0], 1, self.Dx).expand(-1, self.N_sample, -1).reshape(-1, self.Dx)
                
            bm = torchsde.BrownianInterval(t0=self.t_vec_subdivised[0], t1=self.t_vec_subdivised[-1], size=(self.N_sample, self.W_Dx), device = self.t_vec.device)
            
            xhat = torchsde.sdeint(self.sde, x0, self.t_vec_subdivised, method=self.method, bm = bm)
            jacobian_xhat = torch.zeros((len(self.t_vec_subdivised), self.N_sample, self.Dx, self.Dx)).to(self.t_vec)
            hessian_xhat = torch.zeros((len(self.t_vec_subdivised), self.N_sample, self.Dx, self.Dx, self.Dx)).to(self.t_vec)

            for i in range(self.Dx):
                torch_eps = torch.zeros_like(x0)
                torch_eps[:, i] = self.eps
    
                xhat_eps = torchsde.sdeint(self.sde, x0 + torch_eps, self.t_vec_subdivised, method=self.method, bm=bm)
                xhat_neg_eps = torchsde.sdeint(self.sde, x0 - torch_eps, self.t_vec_subdivised, method=self.method, bm=bm)
                
                jacobian_xhat[..., i] = (xhat_eps - xhat_neg_eps) / (2*self.eps)
                
                # partial^2 f / partial x_i ^2
                hessian_xhat[:,:, i, i, :] = torch.clamp((xhat_eps - 2 * xhat + xhat_neg_eps) / (self.eps ** 2), min=-1e10, max=1e10) # derivee seconde
                
            # partial^2 f / partial x_i partial x_j
            torch_full_eps = self.eps * torch.ones_like(x0)
            torch_eps_neg_eps = torch.clone(torch_full_eps)
            torch_eps_neg_eps[:,1] = -self.eps
            
            torch_neg_eps_eps = torch.clone(torch_full_eps)
            torch_eps_neg_eps[:,0] = -self.eps
            
            xhat_eps = torchsde.sdeint(self.sde, x0 + torch_full_eps, self.t_vec_subdivised, method=self.method, bm=bm)
            xhat_neg_eps = torchsde.sdeint(self.sde, x0 - torch_full_eps, self.t_vec_subdivised, method=self.method, bm=bm)
            
            xhat_eps_neg_eps = torchsde.sdeint(self.sde, x0 + torch_eps_neg_eps, self.t_vec_subdivised, method=self.method, bm=bm)
            xhat_neg_eps_eps = torchsde.sdeint(self.sde, x0 + torch_neg_eps_eps, self.t_vec_subdivised, method=self.method, bm=bm)
            
            partial_Xhat_x1_x2 = (xhat_eps - xhat_eps_neg_eps - xhat_neg_eps_eps + xhat_neg_eps)/(4 * (self.eps**2) )
            hessian_xhat[:, :, 0, 1, :] = torch.clamp(partial_Xhat_x1_x2, min = -1e10, max = 1e10)
            hessian_xhat[:, :, 1, 0, :] = torch.clamp(partial_Xhat_x1_x2, min = -1e10, max = 1e10)

        return xhat, jacobian_xhat, hessian_xhat
    

class SdeSampler:
    
    def __init__(self, sde, t_vec, args):
        self.sampler = {
            "inf": ExactSampler,
            "first_order": JacobianSampler,
            "first_order_exact": JacobianSamplerExact,
            "second_order": HessianSampler,
            "second_order_exact": HessianSamplerExact
        }.get(args.sto_taylor_order, None)

        if self.sampler is None:
            raise ValueError("Unsupported method")

        self.sampler = self.sampler(sde, t_vec, args)
        

    def sample(self, x0=None):
        #if x0 is not None:
        #    return self.sampler.sample(x0)
        return self.sampler.sample(x0)



    
def feynman_kac(sde, xhat_samples, x, t, args, jacobian_xhat = None, hessian_xhat = None, x0 = None):
    """
    Compute the Feynman-Kac solution for a given SDE.

    Args:
        sde: The stochastic differential equation instance.
        xhat_samples (Tensor): Sampled SDE paths.
        x (Tensor): Initial conditions.
        t (Tensor): Time vector.
        jacobian_xhat (Tensor, optional): Jacobian of stochastic flow.
        hessian_xhat (Tensor, optional): Hessian of stochastic flow.

    Returns:
        Tensor: The computed Feynman-Kac solution.
    """
    with torch.no_grad():
        if x0 is None:
            x0 = torch.zeros_like(x)
        if args.sto_taylor_order in ["first_order", "first_order_exact"]:
            
            xhat_starting_at_x = xhat_samples + torch.matmul(jacobian_xhat[None,...],x[:,None,None,:,None] - x0[:,None, None,:,None]).squeeze(-1)
        
        elif args.sto_taylor_order == "inf": # exact sampling
            
            xhat_starting_at_x = xhat_samples
        
        elif args.sto_taylor_order in ["second_order", "second_order_exact"]:
            second_order_approx = torch.zeros(args.batch_size, len(t), args.N_sample, args.Dx, device = args.device)
            for i in range(args.Dx):
                second_order_approx[...,i] = torch.sum(x[:,None,None,:]*torch.matmul(hessian_xhat[None,...,i],x[:,None,None,:,None]).squeeze(-1), dim = -1)
            
            xhat_starting_at_x = xhat_samples + torch.matmul(jacobian_xhat[None,...],x[:,None,None,:,None]).squeeze(-1) + 0.5*second_order_approx
                    
        else:
            raise ValueError("Unsupported method")
            
        psi_xhat = sde.density(torch.tensor([0.]).to(x), xhat_starting_at_x, args)
        V_xhat = sde.v(t, xhat_starting_at_x)
        exp_int_Vxhat = torch.exp(-torch.cumulative_trapezoid(V_xhat,x = t, dim = 1))
        
        # concatenate a layer of 1 for the 0-integration between t[0] and t[0]
        exp_int_Vxhat = torch.cat((torch.ones_like(exp_int_Vxhat[:,0:1,:,:]), exp_int_Vxhat), dim = 1)
        
        fc_solution = torch.mean(exp_int_Vxhat*psi_xhat, axis = -2).view(-1, 1)
        
        
        return fc_solution
    
    
    
    
    
    
    