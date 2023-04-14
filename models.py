from locally_connected import LocallyConnected
import torch
import torch.nn as nn
import numpy as np

class mlp_signed(nn.Module):
    # dagma nonlinear
    def __init__(self, dims, bias=True, dtype=torch.float64):
        super(mlp_signed, self).__init__()
        torch.set_default_dtype(torch.double)
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.dims, self.d = dims, dims[0]
        self.I = torch.eye(self.d,dtype=torch.float)
        self.dtype = dtype
        #self.Id = torch.eye(self.d, dtype=torch.float)
        self.fc1 = nn.Linear(self.d, self.d * dims[1], bias=bias)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)
        x = x.squeeze(dim=2)
        return x

    def l1_loss(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(torch.abs(self.fc1.weight))
        return reg

    
    def l2_loss(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        reg += torch.sum(self.fc1.weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    @torch.no_grad()
    def adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)  
        A = torch.sum(fc1_weight ** 2, dim=1).t() 
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()  # [i, j]
        return W


class linear_signed:
    def __init__(self, d, verbose=False, dtype=torch.double):
        super().__init__()
        torch.set_default_dtype(torch.double)
        self.dtype = dtype
        self.d = d
        self.I = torch.eye(d, dtype=torch.float)
        self.vprint = print if verbose else lambda *a, **k: None
        self.W = torch.zeros((d,d),requires_grad=True)


    def forward(self, x):
 
        x = torch.matmul(x, self.W)

        return x

    
    def l1_loss(self):
        """Take l1 norm """
        reg = torch.sum(torch.abs(self.W))
        #reg = sum([p.abs().sum() for p in self.model.parameters()])
        return reg

    
    def l2_loss(self):
        reg = torch.sum((self.W)**2)
        return reg
    
    @torch.no_grad()
    def adj(self):
        W = self.W
       # W = W.cpu().detach().numpy()

        return W 
    


class mlp_unsigned(nn.Module):
    # notears nolinear
    def __init__(self, dims, bias=True):
        super(mlp_unsigned, self).__init__()
        torch.set_default_dtype(torch.double)
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(self.d, self.d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(self.d, self.d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x
    
    def l1_loss(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg
    
    def l2_loss(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    @torch.no_grad()
    def adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


class linear_unsigned:

    def __init__(self, d, verbose=False, dtype=torch.double):
        super().__init__()
        self.dtype = dtype
        self.d = d
        self.vprint = print if verbose else lambda *a, **k: None
        self.w = torch.zeros(2 * d * d)

    def forward(self, x):

        x = torch.matmul(x, self.W)

        return x

    def adj(self):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (self.w[:self.d * self.d] - self.w[self.d * self.d:]).reshape([self.d, self.d])
    
    def l1_loss(self):
        reg = torch.sum(torch.abs(self.W))
        return reg
    
    def l2_loss(self):
        reg = torch.sum((self.W)**2)
        return reg
    
""" 
model = linear_signed(d=10)
model.W = 
model.adj() # this will be the adj

model = nonlinear(d=10)
model.fc1_to_adj
model.adj() # 
"""