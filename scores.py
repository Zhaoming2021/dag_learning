import torch
import torch.nn as nn
import numpy as np
from scipy.special import expit as sigmoid

class LossFunction:
    def __init__(self, model_type, loss_type=None):
        self.model_type = model_type
        self.loss_type = loss_type
        
        if model_type == "mlp":
            if loss_type == 'squared_loss':
                self.loss_fn = self.squared_loss
            elif loss_type == 'log_mse_loss':
                self.loss_fn = self.log_mse_loss
            else:
                raise ValueError('unknown loss type')
        elif model_type == "linear":
            self.loss_fn = self.linear_loss
        else:
            raise ValueError("Invalid model type")
            
    def squared_loss(self, output, target):
        n = target.shape[0]
        loss = 0.5 / n * torch.sum((output - target) ** 2)
        return loss
    
    def log_mse_loss(self, output, target):
        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss
    
    def linear_loss(self, W, X):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if self.loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif self.loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif self.loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss


def main():

    loss_fn = LossFunction(model_type="mlp", loss_type="squared_loss")
    output = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([2.0, 3.0, 4.0])
    loss = loss_fn.loss_fn(output, target)
    print(loss) # should print tensor(0.5000)

if __name__ == '__main__':
    main()
