import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from locally_connected import LocallyConnected
from trace_expm import trace_expm

class MyModel(nn.Module):
    def __init__(self, model, model_type, dims,loss_type, verbose=False, bias=True, dtype=np.float64):
        super(MyModel, self).__init__()
        self.model = model
        self.model_type = model_type

        if model == "linear":

            if model_type == "notears_linear":
                

            elif model_type == "dagma_linear":
                losses = ['l2', 'logistic']
                assert loss_type in losses, f"loss_type should be one of {losses}"
                self.loss_type = loss_type
                self.dtype = dtype
                self.vprint = print if verbose else lambda *a, **k: None

            
        



        elif model == "mlp":
            assert len(dims) >= 2
            assert dims[-1] == 1
            self.dims, self.d = dims, dims[0]
            self.model_type = model_type

            if model_type == "notears_mlp":
                self.fc1_pos = nn.Linear(self.d, self.d * dims[1], bias=bias)
                self.fc1_neg = nn.Linear(self.d, self.d * dims[1], bias=bias)
                self.fc1_pos.weight.bounds = self._bounds()
                self.fc1_neg.weight.bounds = self._bounds()

                layers = []
                for l in range(len(dims) - 2):
                    layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))
                self.fc2 = nn.ModuleList(layers)

            elif model_type == "dagma_mlp":
                self.I = torch.eye(self.d)
                self.fc1 = nn.Linear(self.d, self.d * dims[1], bias=bias)
                nn.init.zeros_(self.fc1.weight)
                nn.init.zeros_(self.fc1.bias)
                layers = []
                for l in range(len(dims) - 2):
                    layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))
                self.fc2 = nn.ModuleList(layers)
        else:
            raise ValueError("Invalid model type")

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
        if self.model_type == "notears_mlp":
            x = self.fc1_pos(x) - self.fc1_neg(x)
            x = x.view(-1, self.dims[0], self.dims[1])
            for fc in self.fc2:
                x = torch.sigmoid(x)
                x = fc(x)
            x = x.squeeze(dim=2)

        elif self.model_type == "dagma_mlp":
            x = self.fc1(x)
            x = x.view(-1, self.dims[0], self.dims[1])
            for fc in self.fc2:
                x = torch.sigmoid(x)
                x = fc(x)
            x = x.squeeze(dim=2)

        return x

    def h_func(self, s=1.0):
        if self.model_type == "notears_mlp":
            d = self.dims[0]
            fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight
            fc1_weight = fc1_weight.view(d, -1, d)
            A = torch.sum(fc1_weight * fc1_weight, dim=1).t()
            h = trace_expm(A) - d
            return h

        elif self.model_type == "dagma_mlp":
            fc1_weight = self.fc1.weight
            fc1_weight = fc1_weight.view(self.d, -1, self.d)
            A = torch.sum(fc1_weight ** 2, dim=1).t()
            h = -torch.slogdet(s * self.I - A)[1] + self.d * np.log(s)
            return h

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
        return torch.sum(torch.abs(self.fc1.weight))

    def fc1_to_adj(self) -> np.ndarray:
        if self.model_type == "notears_mlp":
            d = self.dims[0]
            fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight
            fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
            A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
            W = torch.sqrt(A)  # [i, j]
            W = W.cpu().detach().numpy()
            
        elif self.model_type == "dagma_mlp":
            fc1_weight = self.fc1.weight
            fc1_weight = fc1_weight.view(self.d, -1, self.d)  
            A = torch.sum(fc1_weight ** 2, dim=1).t() 
            W = torch.sqrt(A)
            W = W.cpu().detach().numpy()  # [i, j]

        return W


