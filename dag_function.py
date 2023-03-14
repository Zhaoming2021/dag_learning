# notears.py
# dagma.py
# poly.py
import torch
import torch.nn as nn
import numpy as np
from trace_expm import trace_expm


class dag_functions:
    def __init__(self, model, X, W, dims, s=1.0):
        self.model = model
        self.X = X
        self.W = W
        self.dims, self.d = dims, dims[0]
        self.s = s
        self.I = torch.eye(self.d)

    def h(self):
        if self.model.type == "mlp_signed":
            #similar to h in dagma nonlinear
            fc1_weight = self.fc1.weight
            fc1_weight = fc1_weight.view(self.d, -1, self.d)
            A = torch.sum(fc1_weight ** 2, dim=1).t()
            h = -torch.slogdet(self.s * self.I - A)[1] + self.d * np.log(self.s)
            return h

        elif self.model.type == "linear_signed":
            # similar to h in dagma linear
            M = self.s * self.Id - self.W @ self.W
            h = - torch.slogdet(M)[1] + self.dims[0] * np.log(self.s)
            G_h = 2 * self.W @ torch.inverse(M).T 
            return h, G_h

        elif self.model.type == "mlp_unsigned":
            #similar to h in notears nonlinear
            d = self.dims[0]
            fc1_pos_weight = self.model.fc1_pos.weight
            fc1_neg_weight = self.model.fc1_neg.weight
            fc1_weight = fc1_pos_weight - fc1_neg_weight
            fc1_weight = fc1_weight.view(d, -1, d)
            A = torch.sum(fc1_weight * fc1_weight, dim=1).t()
            h = trace_expm(A) - d
            return h
        elif self.model.type == "linear_unsigned":
            # similar to h in notears linear
            
            """Evaluate value and gradient of acyclicity constraint."""
            n, d = self.X.shape
            E = torch.matrix_exp(self.W @ self.W)  # (Zheng et al. 2018)
            h = torch.trace(E) - d
            G_h = torch.matmul(torch.transpose(E, 0, 1), self.W) * 2
            return h, G_h
