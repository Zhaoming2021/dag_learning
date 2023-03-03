import torch
import torch.nn as nn
import numpy as np
from scipy.special import expit as sigmoid

class LossFunction:
    def __init__(self, model, loss_type=None):
        self.model_type = model
        self.loss_type = loss_type

        def squared_loss(self, output, target):
            n = target.shape[0]
            loss = 0.5 / n * torch.sum((output - target) ** 2)
            return loss
        def log_mse_loss(self, output, target):
            n, d = target.shape
            loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
            return loss

        def linear_loss(W, X, loss_type):
            """Evaluate value and gradient of loss."""
            X_tensor = torch.tensor(X, dtype=torch.float32)
            W_tensor = torch.tensor(W, dtype=torch.float32, requires_grad=True)
            M = X_tensor @ W_tensor
            if loss_type == 'l2':
                R = X_tensor - M
                loss = 0.5 / X_tensor.shape[0] * (R ** 2).sum()
            elif loss_type == 'logistic':
                loss = 1.0 / X_tensor.shape[0] * (torch.logaddexp(torch.tensor(0.), M) - X_tensor * M).sum()
            elif loss_type == 'poisson':
                S = torch.exp(M)
                loss = 1.0 / X_tensor.shape[0] * (S - X_tensor * M).sum()
            else:
                raise ValueError('unknown loss type')
            loss.backward()
            G_loss = W_tensor.grad.detach().numpy()
            return loss.detach().numpy(), G_loss


