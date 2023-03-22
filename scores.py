import torch
import torch.nn as nn
import numpy as np
from scipy.special import expit as sigmoid

class LossFunction:
    def squared_loss(self, output, target):
        n = target.shape[0]
        loss = 0.5 / n * torch.sum((output - target) ** 2)
        return loss

    def log_mse_loss(self, output, target):
        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss
    
    def logistic_loss(self, output, target):
        #X_tensor = torch.tensor(X, dtype=torch.double)
        #W_tensor = torch.tensor(W, dtype=torch.double, requires_grad=True)
        #M = target @ output
        #loss = 1.0 / target.shape[0] * (torch.logaddexp(torch.tensor(0.), M) - target * M).sum()
        loss = 1.0 / output.shape[0] * (torch.logaddexp(torch.tensor(0.), target) - target * output).sum()
        # Compute the gradients
        #G_loss = loss.backward()
        return loss
    
    def poisson_loss(self, output, target):
        #M = target @ output
        S = torch.exp(target)
        loss = 1.0 / output.shape[0] * (S - output * target).sum()
        # Compute the gradients
        #G_loss = loss.backward()
        return loss
    """  
    def poisson_loss(self, output, target):
        M = target @ output
        S = torch.exp(M)
        loss = 1.0 / target.shape[0] * (S - target * M).sum()
        loss.backward()
        return loss
    
    def linear_loss(self, W, X, loss_type):
        X_tensor = torch.tensor(X, dtype=torch.double)
        W_tensor = torch.tensor(W, dtype=torch.double, requires_grad=True)
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
    """
def main():
    loss_fn = LossFunction()
    output = torch.tensor([[1, 2, 3],[2, 4, 6]], dtype=torch.double)
    target = torch.tensor([[2, 4, 6],[1, 2, 3]], dtype=torch.double)

    squared_loss = loss_fn.squared_loss(output, target)
    log_mse_loss = loss_fn.log_mse_loss(output, target)

if __name__ == '__main__':
    main()
#loss_fn = Loss()
#score = LossFunction.squared_loss(output, target)

#score = mse_loss(data, model, hyperparams) 