from locally_connected import LocallyConnected
import torch
import torch.nn as nn
import numpy as np

class mlp_signed(nn.Module):
    # dagma nonlinear
    def __init__(self, dims, bias=True, dtype=torch.float64):
        super(mlp_signed, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.dims, self.d = dims, dims[0]
        self.I = torch.eye(self.d)
        self.dtype = dtype
        #self.Id = np.eye(self.d).astype(self.dtype)
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

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        return torch.sum(torch.abs(self.fc1.weight))

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)  
        A = torch.sum(fc1_weight ** 2, dim=1).t() 
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()  # [i, j]
        return W
    
""" 
class linear_signed:
    # dagma linear
    def __init__(self, d, verbose=False,s = 1.0, dtype=torch.double):
        super().__init__()
        self.dtype = dtype
        self.s = s
        self.d = d
        self.vprint = print if verbose else lambda *a, **k: None
        self.Id = torch.eye(self.d).astype(self.dtype)
        self.W = torch.zeros((self.d,self.d)).astype(self.dtype) # init W0 at zero matrix

    def _adam_update(self, grad, iter, beta_1, beta_2):
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iter)
        v_hat = self.opt_v / (1 - beta_2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)

        return grad
    def adj(self):
        M = torch.linalg.inv(self.s * self.Id - self.W * self.W) + 1e-16
        Gobj = G_score + mu * self.lambda1 * torch.sign(W) + 2 * W * M.T
        ## Adam step
        grad = self._adam_update(Gobj, iter, self.beta_1, self.beta_2)
        W -= lr * grad    
        #W = torch.from_numpy(W)
        return W 
"""

    
class mlp_unsigned(nn.Module):
    # notears nolinear
    def __init__(self, dims, bias=True):
        super(mlp_unsigned, self).__init__()
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
    
    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
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
        #self.w = np.zeros(2 * d * d)
        self.w = torch.zeros(2 * d * d)
        #self.W = torch.zeros((d,d))

    def _adj(self):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (self.w[:self.d * self.d] - self.w[self.d * self.d:]).reshape([self.d, self.d])

    def adj(self):
        W = self._adj()
        #W = torch.from_numpy(W)
        return W
    
""" 
model = linear_signed(d=10)
model.W = 
model.adj() # this will be the adj

model = nonlinear(d=10)
model.fc1_to_adj
model.adj() # 
"""