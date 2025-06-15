import torch
from torch import nn
from torch.nn import functional as F
import math


def default_initializer(size, std=0.02):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.randn(size=size, dtype=torch.float32, device=device) * std

class W_LU(nn.Module):
    def __init__(self, input_dim):
        super(W_LU, self).__init__()
        self.input_dim = input_dim
        self.LU = nn.Parameter(data=torch.zeros(size=[self.input_dim, self.input_dim], dtype=torch.float32), requires_grad=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # identity matrix
        self.LU_init = torch.eye(self.input_dim, dtype=torch.float32).to(self.device)

    def forward(self, inputs, logdet=None, reverse=False):
        x = inputs
        LU = self.LU_init + self.LU

        # upper-triangular matrix
        U = torch.triu(LU)

        # diagonal line
        U_diag = torch.diag(U)

        I = torch.eye(self.input_dim, dtype=torch.float32).to(self.device)
        L = torch.tril(I + LU) - torch.diag(torch.diag(LU))  # 对角线元素为1

        if not reverse:
            x = x.T
            x = U @ x
            x = L @ x
            x = x.T
        else:
            x = x.T
            x = torch.inverse(L) @ x
            x = torch.inverse(U) @ x

            x = x.T

        if logdet is not None:
            dlogdet = torch.sum(torch.log(torch.abs(U_diag)))
            if reverse:
                dlogdet *= -1.0
            return x, logdet + dlogdet

        return x





# one linear layer with default width 32.
class Linear(nn.Module):
    def __init__(self, input_dim=2, n_width=32):
        super(Linear, self).__init__()
        self.input_dim = input_dim
        self.n_width = n_width
        #self.w = nn.Parameter(
        #    data=default_initializer(size=[self.input_dim, self.n_width]), requires_grad=True)
        self.w = nn.Parameter(
            data=torch.rand(size=[self.input_dim, self.n_width], dtype = torch.float32), requires_grad=True)
        
        self.b = nn.Parameter(
            data=torch.zeros(size=[self.n_width, ], dtype=torch.float32), requires_grad=True)

    def forward(self, inputs):

        return inputs @ self.w + self.b


# two-hidden-layer neural network
class NN2(nn.Module):
    def __init__(self, n_width, n_in, n_out):
        super(NN2, self).__init__()
        self.n_width = n_width
        self.n_out = n_out
        self.n_in = n_in
        #self.l_1 = Linear(input_dim=self.n_in, n_width=self.n_width)
        #self.l_2 = Linear(input_dim=self.n_width, n_width=self.n_width)
        #self.l_f = Linear(input_dim=self.n_width, n_width=self.n_out)
        self.l_1 = nn.Linear(self.n_in, self.n_width)
        self.l_2 = nn.Linear(self.n_width, self.n_width)
        self.l_f = nn.Linear(self.n_width, self.n_out)
        

    def forward(self, inputs):

        # relu with low regularity
        # x = F.softplus(self.l_1(inputs))
        # x = F.softplus(self.l_2(x))
        x = torch.tanh(self.l_1(inputs))
        x = torch.tanh(self.l_2(x))
        x = self.l_f(x)

        return x


# four-hidden-layer neural network
class NN2v(nn.Module):
    def __init__(self, n_width=32, n_in=2, n_out=2):
        super(NN2v, self).__init__()
        self.n_width = n_width
        self.n_out = n_out
        self.n_in = n_in
        self.l_1 = Linear(input_dim=self.n_in, n_width=self.n_width)
        self.l_2 = Linear(input_dim=self.n_width, n_width=self.n_width // 2)
        self.l_3 = Linear(input_dim=self.n_width // 2, n_width=self.n_width // 2)
        self.l_4 = Linear(input_dim=self.n_width // 2, n_width=self.n_width)

        self.l_f = Linear(input_dim=self.n_width // 2, n_width=self.n_out)

    def forward(self, inputs):
        # relu with low regularity

        # x = F.softplus(self.l_1(inputs))
        # x = F.softplus(self.l_2(x))
        # x = F.softplus(self.l_3(x))
        # x = F.softplus(self.l_4(x))
        x = torch.tanh(self.l_1(inputs))
        x = torch.tanh(self.l_2(x))
        x = torch.tanh(self.l_3(x))
        x = torch.tanh(self.l_4(x))

        x = self.l_f(x)

        return x


# affine coupling layer
class affine_coupling(nn.Module):
    def __init__(self, n_split_at=1, cond_dim=None, flag=0, input_dim=2, n_width=32, flow_coupling=1):
        super(affine_coupling, self).__init__()
        # partition as [:n_split_at] and [n_split_at:]
        if cond_dim is not None:
            total_input_dim = input_dim + cond_dim
        else:
            total_input_dim = input_dim
        self.n_split_at = n_split_at
        self.flow_coupling = flow_coupling
        self.n_width = n_width
        self.flag = flag
        self.input_dim = input_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.flag > 0:
            # fix z1, change z2
            self.f_input_dim = total_input_dim - self.input_dim + self.n_split_at
        else:
            # fix z2, change z1
            self.f_input_dim = total_input_dim - self.n_split_at
        if self.flow_coupling == 0:
            # no scaling
            self.f = NN2(self.n_width, self.f_input_dim, total_input_dim - self.f_input_dim)
        elif self.flow_coupling == 1:
            # with scaling
            self.f = NN2(self.n_width, self.f_input_dim, (total_input_dim - self.f_input_dim) * 2)
        else:
            raise Exception()
        self.log_gamma = nn.Parameter(data=torch.zeros(size=[1, total_input_dim - self.f_input_dim],
                                      dtype=torch.float32, device=self.device), requires_grad=True)

    def forward(self, inputs, cond_t=None, logdet=None, reverse=False):
        z = inputs

        n_split_at = self.n_split_at

        alpha = 0.6
        if self.flag:
            z1 = z[..., :n_split_at]
            z2 = z[..., n_split_at:]
        else:
            z2 = z[..., :n_split_at]
            z1 = z[..., n_split_at:]
        z1_m = z1
        if cond_t is not None:
            z1_m = torch.cat([z1, cond_t], -1)
        if not reverse:

            if self.flow_coupling == 0:  # do not scaling
                shift = self.f(z1_m)
                shift = torch.exp(self.log_gamma) * torch.tanh(shift)
                z2 += shift
            elif self.flow_coupling == 1:  # with scaling
                h = self.f(z1_m)
                shift = h[..., 0::2]

                scale = alpha * torch.tanh(h[..., 1::2])
                shift = torch.exp(self.log_gamma) * torch.tanh(shift)
                z2 = z2 + scale * z2 + shift
                if logdet is not None:
                    logdet += torch.sum(torch.log(scale + torch.ones_like(scale)), -1, keepdim=True)
            else:
                raise Exception()

        else:

            if self.flow_coupling == 0:
                shift = self.f(z1_m)
                shift = torch.exp(self.log_gamma) * torch.tanh(shift)
                z2 -= shift
            elif self.flow_coupling == 1:
                h = self.f(z1_m)
                shift = h[..., 0::2]

                scale = alpha * torch.tanh(h[..., 1::2])
                shift = torch.exp(self.log_gamma) * torch.tanh(shift)
                z2 = (z2 - shift) / (torch.ones_like(scale) + scale)
                if logdet is not None:
                    logdet -= torch.sum(torch.log(scale + torch.ones_like(scale)),
                                        -1, keepdim=True)
            else:
                raise Exception()
        if self.flag:
            z = torch.cat([z1, z2], -1)
        else:
            z = torch.cat([z2, z1], -1)

        if logdet is not None:
            return z, logdet

        return z

class affine_coupling_1d(nn.Module):
    def __init__(self, cond_dim=None, n_width=32, input_dim = 1):
        super(affine_coupling_1d, self).__init__()
        
        if cond_dim is not None:
            total_input_dim = input_dim + cond_dim
        else:
            total_input_dim = input_dim
        
        self.n_width = n_width
        
        self.input_dim = input_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.f_input_dim = total_input_dim
        
        self.f = NN2(self.n_width, self.f_input_dim, input_dim + 1)
        
        self.log_gamma = nn.Parameter(data=torch.zeros(size=[1, input_dim],
                                      dtype=torch.float32, device=self.device), requires_grad=True)

    def forward(self, inputs, cond_t=None, logdet=None, reverse=False):
        z = inputs

        alpha = 0.6
        
        z1_m = z
        if cond_t is not None:
            z1_m = torch.cat([z, cond_t], -1)
        if not reverse:

            h = self.f(z1_m)
            s = h[..., 0:1]
            q = h[..., :1]

            scale = alpha * torch.tanh(s)
            shift = torch.exp(self.log_gamma) * torch.tanh(q)
            z = z + scale * z + shift
            
            if logdet is not None:
                logdet += torch.sum(torch.log(scale + torch.ones_like(scale)), -1, keepdim=True)

        else:
            
            h = self.f(z1_m)
            s = h[..., 0:1]
            q = h[..., :1]

            scale = alpha * torch.tanh(s)
            shift = torch.exp(self.log_gamma) * torch.tanh(q)
            z = (z - shift) / (torch.ones_like(scale) + scale)
            if logdet is not None:
                logdet -= torch.sum(torch.log(scale + torch.ones_like(scale)), -1, keepdim=True)
                
        if logdet is not None:
            return z, logdet

        return z
    
class MLP_mapping(nn.Module):
    def __init__(self, input_dim = 2, n_depth = 4, n_width = 32):
        super(MLP_mapping, self).__init__()
        self.n_depth = n_depth
        self.n_width = n_width
        self.input_dim = input_dim + 1 # +1 for the temporal dimension
        self.output_dim = 1 # probability density

        # Define layers
        layers = []
        layers.append(nn.Linear(self.input_dim, self.n_width))

        for _ in range(self.n_depth):
            layers.append(nn.Linear(self.n_width, self.n_width))
        
        layers.append(nn.Linear(self.n_width, self.output_dim))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)  # output layer, no activation
    
class flow_mapping(nn.Module):
    def __init__(self, input_dim=2, n_depth=4, n_split_at=1, cond_dim=None, n_width=32, flow_coupling=1, n_bins=0, rotation=True):
        super(flow_mapping, self).__init__()
        self.n_depth = n_depth
        self.n_split_at = n_split_at
        self.n_width = n_width
        self.flow_coupling = flow_coupling
        self.n_bins = n_bins

        self.input_dim = input_dim
        self.scale_layers = []
        self.affine_layers = []
        self.rotation = rotation
        if self.rotation is True:
            self.W_LU_layers = W_LU(input_dim=self.input_dim)

        if self.n_bins > 0:
            self.nonlinear_layers = CDF_quadratic(self.input_dim, self.n_bins)
        sign = -1
        for i in range(self.n_depth):
            
            self.scale_layers.append(actnorm(self.input_dim))
            sign *= -1
            
            i_split_at = (self.n_split_at * sign + self.input_dim) % self.input_dim
            
            if input_dim > 1:
                self.affine_layers.append(affine_coupling(i_split_at, cond_dim=cond_dim,
                                                          n_width=self.n_width,
                                                          flow_coupling=self.flow_coupling, flag=(sign > 0), input_dim=self.input_dim))
            elif input_dim == 1:
                self.affine_layers.append(affine_coupling_1d(cond_dim=cond_dim, n_width=self.n_width, input_dim = input_dim))
                
        self.scale_layers = nn.ModuleList(self.scale_layers)
        self.affine_layers = nn.ModuleList(self.affine_layers)

        assert n_depth % 2 == 0

    # without computing the jacobian.
    def forward(self, inputs, cond_t=None, logdet=None, reverse=False):

        if not reverse:
            z = inputs
            if self.rotation is True:
                if logdet is not None:
                    z, logdet = self.W_LU_layers(z, logdet)
                else:
                    z = self.W_LU_layers(z)

            for i in range(self.n_depth):

                if logdet is not None:
                    z, logdet = self.scale_layers[i](z, logdet)
                else:
                    z = self.scale_layers[i](z)

                z = self.affine_layers[i](z, cond_t, logdet)
                if logdet is not None:
                    z, logdet = z
            if self.n_bins > 0:
                if logdet is not None:
                    z, logdet = self.nonlinear_layers(z, logdet)
                else:
                    z = self.nonlinear_layers(z)

        else:
            z = inputs
            if self.n_bins > 0:
                if logdet is not None:
                    z, logdet = self.nonlinear_layers(z, logdet, reverse=True)
                else:
                    z = self.nonlinear_layers(z, reverse=True)

            for i in reversed(range(self.n_depth)):
                # z = z[:,::-1]

                z = self.affine_layers[i](z, cond_t, logdet, reverse=True)
                if logdet is not None:
                    z, logdet = z

                z = self.scale_layers[i](z, logdet, reverse=True)
                if logdet is not None:
                    z, logdet = z
            if self.rotation is True:
                if logdet is not None:
                    z, logdet = self.W_LU_layers(z, logdet, reverse=True)
                else:
                    z = self.W_LU_layers(z, reverse=True)

        if logdet is not None:
            return z, logdet
        return z


class actnorm(nn.Module):
    def __init__(self, input_dim=2, scale=1.0, logscale_factor=3.0):
        super(actnorm, self).__init__()
        '''
                input_dim: input dimension 
        '''
        self.scale = scale
        self.logscale_factor = logscale_factor
        self.input_dim = input_dim
        self.data_init = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.b = nn.Parameter(data=torch.zeros(size=[1, self.input_dim], dtype=torch.float32), requires_grad=True)
        self.b_init = (torch.zeros(size=[1, self.input_dim], dtype=torch.float32)).to(self.device)
        self.logs = nn.Parameter(data=torch.zeros(size=[1, self.input_dim], dtype=torch.float32), requires_grad=True)
        self.logs_init = (torch.zeros(size=[1, self.input_dim], dtype=torch.float32)).to(self.device)

    def forward(self, inputs, logdet=None, reverse=False):
        assert self.input_dim == inputs.shape[-1], 'please check your input dimension!'

        if not self.data_init:
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x_mean = torch.mean(inputs, 0, keepdim=True)
            x_var = torch.mean(torch.square(inputs - x_mean), 0, keepdim=True)
            x_mean = x_mean.to(self.device)
            x_var = x_var.to(self.device)
            self.b_init.data.add(-x_mean)
            self.logs_init.data.add(torch.log(self.scale / (torch.sqrt(x_var) + 1e-6)) / self.logscale_factor)

            self.data_init = True

        if not reverse:
            x = inputs + (self.b + self.b_init)
            x = x * torch.exp(self.logs + self.logs_init)
        else:
            x = inputs * torch.exp(-self.logs - self.logs_init)
            x = x - (self.b + self.b_init)

        if logdet is not None:
            dlogdet = torch.sum(self.logs + self.logs_init) * torch.ones_like(logdet)
            if reverse:
                dlogdet *= -1
            return x, logdet + dlogdet

        return x

class ActNorm_git(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(input_dim))
        self.bias = torch.nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("_initialized", torch.tensor(False))

    def reset_(self):
        self._initialized = torch.tensor(False)
        return self

    def forward(self, x, logdet=None, reverse=False):
        #self._check_input_dim(x)
        if x.dim() > 2:
            x = x.transpose(1, -1)
        if not self._initialized:
            self.scale.data = 1 / x.detach().reshape(-1, x.shape[-1]).std(
                0, unbiased=False
            )
            self.bias.data = -self.scale * x.detach().reshape(
                -1, x.shape[-1]
            ).mean(0)
            self._initialized = torch.tensor(True)
        
        if not reverse : 
            x = self.scale * x + self.bias
        else:
            x = (x - self.bias)/self.scale
        
        if logdet is not None:
            dlogdet = torch.log(torch.sum(self.scale))
            if reverse:
                dlogdet *= -1
            return x, logdet + dlogdet
        
        return x
    

################### inutile
##############################
class CDF_quadratic(nn.Module):
    def __init__(self, input_dim, n_bins, r=1.2, bound=10.0, beta=1e-6):
        super(CDF_quadratic, self).__init__()

        assert n_bins % 2 == 0

        self.n_bins = n_bins
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # generate a nonuniform mesh symmetric to zero,
        # and increasing by ratio r away from zero.
        self.bound = bound
        self.r = r
        self.beta = beta

        m = n_bins / 2
        x1L = bound * (r - 1.0) / (math.pow(r, m) - 1.0)

        index = torch.arange(0, self.n_bins + 1, dtype=torch.float32).reshape(-1, 1)
        index = index - m
        xr = (1. - torch.pow(r, torch.abs(index))) / (1. - r)
        xr = torch.where(index >= 0, x1L * xr, -x1L * xr)
        xr = xr.reshape(-1, 1)
        xr = ((xr + bound) / 2.0 / bound).to(self.device)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.x1L = (x1L / 2.0 / bound)
        self.mesh = torch.cat((torch.tensor(0.0, device=self.device).reshape(-1, 1), xr[1:-1, 0].reshape(-1, 1),
                                torch.tensor(1.0, device=self.device).reshape(-1, 1)), 0)
        self.elmt_size = (self.mesh[1:] - self.mesh[:-1]).reshape(-1, 1)
        self.input_dim = input_dim
        self.p = nn.Parameter(data=torch.zeros(size=[self.n_bins - 1, self.input_dim], dtype=torch.float32,
                                                device=self.device), requires_grad=True)

    def forward(self, inputs, logdet=None, reverse=False):

        # normalize the PDF
        self._pdf_normalize()

        x = inputs
        if not reverse:
            # for the interval [-a,a]
            # rescale such points in [-bound, bound] will be mapped to [0,1]
            x = (x + self.bound) / 2.0 / self.bound

            # cdf mapping
            x = self._cdf(x, logdet)
            if logdet is not None:
                x, logdet = x

            # maps [0,1] back to [-bound, bound]
            x = x * 2.0 * self.bound - self.bound

            # for the interval (a,inf)
            x = torch.where(x > self.bound, self.beta * (x - self.bound) + self.bound, x)
            if logdet is not None:
                dlognet = x
                dlogdet = torch.where(dlognet > self.bound, torch.tensor(self.beta, device=self.device),
                                      torch.tensor(1.0, device=self.device))
                dlogdet = torch.sum(torch.log(dlogdet), -1, keepdim=True)
                logdet += dlogdet

            # for the interval (-inf,a)
            x = torch.where(x < -self.bound, self.beta * (x + self.bound) - self.bound, x)
            if logdet is not None:
                dlognet = x
                dlogdet = torch.where(dlognet < -self.bound, torch.tensor(self.beta, device=self.device),
                                      torch.tensor(1.0, device=self.device))
                dlogdet = torch.sum(torch.log(dlogdet), -1, keepdim=True)
                logdet += dlogdet
        else:
            # for the interval [-a,a]
            # rescale such points in [-bound, bound] will be mapped to [0,1]
            x = (x + self.bound) / 2.0 / self.bound

            # cdf mapping
            x = self._cdf_inv(x, logdet)
            if logdet is not None:
                x, logdet = x

            # maps [0,1] back to [-bound, bound]
            x = x * 2.0 * self.bound - self.bound

            # for the interval (a,inf)
            x = torch.where(x > self.bound, (x - self.bound) / self.beta + self.bound, x)
            x = x.to(self.device)
            if logdet is not None:
                dlognet = x
                dlogdet = torch.where(dlognet > self.bound, torch.tensor(1.0 / self.beta, device=self.device),
                                      torch.tensor(1.0, device=self.device))
                dlogdet = torch.sum(torch.log(dlogdet), [1], keepdim=True)
                logdet += dlogdet

            # for the interval (-inf,a)
            x = torch.where(x < -self.bound, (x + self.bound) / self.beta - self.bound, x)
            x = x.to(self.device)
            if logdet is not None:
                dlognet = x
                dlogdet = torch.where(dlognet < -self.bound, torch.tensor(1.0 / self.beta, device=self.device),
                                      torch.tensor(1.0, device=self.device))
                dlogdet = torch.sum(torch.log(dlogdet), [1], keepdim=True)
                logdet += dlogdet

        if logdet is not None:
            return x, logdet

        return x

    # normalize the piecewise representation of pdf
    def _pdf_normalize(self):
        # piecewise pdf
        p0 = torch.ones((1, self.input_dim), dtype=torch.float32, device=self.device) * self.beta
        self.pdf = p0
        px = torch.exp(self.p) * (self.elmt_size[:-1] + self.elmt_size[1:]) / 2.0
        px = (1.0 - (self.elmt_size[0] + self.elmt_size[-1]) * self.beta / 2.0) / torch.sum(px, 0, keepdim=True)
        px = px * torch.exp(self.p)
        self.pdf = torch.cat((self.pdf, px), 0)
        self.pdf = torch.cat((self.pdf, p0), 0)

        # probability in each element
        cell = (self.pdf[:-1, :] + self.pdf[1:, :]) / 2.0 * self.elmt_size
        # CDF - contribution from previous elements.
        r_zeros = torch.zeros((1, self.input_dim), dtype=torch.float32, device=self.device)
        self.F_ref = r_zeros
        for i in range(1, self.n_bins):
            tp = torch.sum(cell[:i, :], 0, keepdim=True)
            self.F_ref = torch.cat((self.F_ref, tp), 0)

    # the cdf is a piecewise quadratic function.
    def _cdf(self, x, logdet=None):
        xr = self.mesh.repeat(1, self.input_dim)
        k_ind = torch.searchsorted(xr.T.contiguous(), x.T.contiguous(), right=True)
        k_ind = torch.t(k_ind)
        # k_ind = torch.tensor(k_ind, dtype=torch.int64)
        k_ind = k_ind.clone().detach()
        k_ind -= 1

        cover = torch.where(k_ind * (k_ind - self.n_bins + 1) <= 0, torch.tensor(1.0, device=self.device),
                            torch.tensor(0.0, device=self.device))

        k_ind = torch.where(k_ind < 0, torch.tensor(0, device=self.device), k_ind)
        k_ind = torch.where(k_ind > (self.n_bins - 1), self.n_bins - 1, k_ind)

        v1 = (self.pdf[:, 0][k_ind[:, 0]]).reshape(-1, 1)
        for i in range(1, self.input_dim):
            tp = (self.pdf[:, i][k_ind[:, i]]).reshape(-1, 1)
            v1 = torch.cat([v1, tp], 1)

        v2 = (self.pdf[:, 0][k_ind[:, 0] + 1]).reshape(-1, 1)
        for i in range(1, self.input_dim):
            tp = (self.pdf[:, i][k_ind[:, i] + 1]).reshape(-1, 1)
            v2 = torch.cat([v2, tp], 1)

        xmodi = torch.reshape(x[:, 0] - self.mesh[:, 0][k_ind[:, 0]], (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(x[:, i] - (self.mesh[:, 0][k_ind[:, i]]), (-1, 1))
            xmodi = torch.cat([xmodi, tp], 1)

        h_list = torch.reshape((self.elmt_size[:, 0][k_ind[:, 0]]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(self.elmt_size[:, 0][k_ind[:, i]], (-1, 1))
            h_list = torch.cat([h_list, tp], 1)

        F_pre = torch.reshape((self.F_ref[:, 0][k_ind[:, 0]]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape((self.F_ref[:, i][k_ind[:, i]]), (-1, 1))
            F_pre = torch.cat([F_pre, tp], 1)

        y = torch.where(cover > 0, F_pre + xmodi ** 2 / 2.0 * (v2 - v1) / h_list + xmodi * v1, x)

        if logdet is not None:
            dlogdet = torch.where(cover > 0, xmodi * (v2 - v1) / h_list + v1, torch.tensor(1.0, device=self.device))
            dlogdet = torch.sum(torch.log(dlogdet), [1], keepdim=True)
            return y, logdet + dlogdet

        return y

    # inverse of the cdf
    def _cdf_inv(self, y, logdet=None):
        xr = self.mesh.repeat(1, self.input_dim)
        yr1 = self._cdf(xr)

        p0 = torch.zeros((1, self.input_dim), dtype=torch.float32, device=self.device)
        p1 = torch.ones((1, self.input_dim), dtype=torch.float32, device=self.device)
        yr = torch.cat([p0, yr1[1:-1, :], p1], 0)

        k_ind = torch.searchsorted(yr.T.contiguous(), y.T.contiguous(), right=True)
        k_ind = k_ind.T
        # k_ind = torch.tensor(k_ind, dtype=torch.int64)
        k_ind = k_ind.clone().detach()

        k_ind -= 1

        cover = torch.where(k_ind * (k_ind - self.n_bins + 1) <= 0, torch.tensor(1.0, device=self.device),
                            torch.tensor(0.0, device=self.device))

        k_ind = torch.where(k_ind < 0, torch.tensor(0, device=self.device), k_ind)
        k_ind = torch.where(k_ind > (self.n_bins - 1), self.n_bins - 1, k_ind)

        c_cover = torch.reshape(cover[:, 0], (-1, 1))
        v1 = torch.where(c_cover > 0, torch.reshape((self.pdf[:, 0][k_ind[:, 0]]), (-1, 1)), -torch.ones_like(c_cover))
        for i in range(1, self.input_dim):
            c_cover = torch.reshape(cover[:, i], (-1, 1))
            tp = torch.where(c_cover > 0, torch.reshape((self.pdf[:, i][k_ind[:, i]]), (-1, 1)), -torch.ones_like(c_cover))
            v1 = torch.cat([v1, tp], 1)

        c_cover = torch.reshape(cover[:, 0], (-1, 1))
        v2 = torch.where(c_cover > 0, torch.reshape((self.pdf[:, 0][k_ind[:, 0] + 1]), (-1, 1)), -2.*torch.ones_like(c_cover))
        for i in range(1, self.input_dim):
            c_cover = torch.reshape(cover[:, i], (-1, 1))
            tp = torch.where(c_cover > 0, torch.reshape((self.pdf[:, i][k_ind[:, i] + 1]), (-1, 1)), -2.0 * torch.ones_like(c_cover))
            v2 = torch.cat([v2, tp], 1)

        ys = torch.reshape(y[:, 0] - (yr[:, 0][k_ind[:, 0]]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(y[:, i] - (yr[:, i][k_ind[:, i]]), (-1, 1))
            ys = torch.cat([ys, tp], 1)

        xs = torch.reshape((xr[:, 0][k_ind[:, 0]]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape((xr[:, i][k_ind[:, i]]), (-1, 1))
            xs = torch.cat([xs, tp], 1)

        h_list = torch.reshape((self.elmt_size[:, 0][k_ind[:, 0]]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape((self.elmt_size[:, 0][k_ind[:, i]]), (-1, 1))
            h_list = torch.cat([h_list, tp], 1)

        h_s = (v2 - v1) / h_list
        tp = v1 * v1 + 2.0 * ys * h_s
        tp = torch.sqrt(tp) + v1
        tp = 2.0 * ys / tp
        tp += xs

        x = torch.where(cover > 0, tp, y)

        if logdet is not None:
            tp = 2.0 * ys * h_s
            tp += v1 * v1
            tp = 1.0 / torch.sqrt(tp)

            dlogdet = torch.where(cover > 0, tp, torch.tensor(1.0, device=self.device))
            dlogdet = torch.sum(torch.log(dlogdet), [1], keepdim=True)
            return x, logdet + dlogdet

        return x