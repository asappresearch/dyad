import torch
import numpy as np

class Dyad(torch.nn.Module):
    def __init__(self,shape,bias=True):
        super().__init__()
        dyad_dim, dim_in, dim_out = shape
        self.has_bias = bias
        k = 1.0/float(np.sqrt(dim_in*dyad_dim))
        self.wu = torch.nn.Parameter(torch.empty((dyad_dim,dim_out,dim_in),dtype=torch.float32))
        torch.nn.init.uniform_(self.wu,-k,k)
        self.wl = torch.nn.Parameter(torch.empty((dyad_dim,dim_out,dim_in),dtype=torch.float32))
        torch.nn.init.uniform_(self.wl,-k,k)
        if self.has_bias:
            self.bias = torch.nn.Parameter(torch.empty((dyad_dim*dim_out,1),dtype=torch.float32))
            torch.nn.init.uniform_(self.bias,-k,k)
        self.dyad_dim = dyad_dim
        self.dim_in = dim_in
        self.dim_out = dim_out
    @torch.autocast("cuda")
    def forward(self,x):
        x1 = x.reshape(self.dyad_dim,self.dim_in,-1)
        x2 = x.reshape(self.dim_in,self.dyad_dim,-1).transpose(0,1)
        out = (self.wl.bmm(x1)+self.wu.bmm(x2)).reshape(self.dyad_dim*self.dim_out,-1)
        if self.has_bias:
            out+= self.bias
        return out
    def extra_repr(self) -> str:
        return f'dyad_dim={self.dyad_dim}, dim_in={self.dim_in}, dim_out={self.dim_out}, bias={self.has_bias}'