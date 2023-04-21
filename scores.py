import torch

def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss

def log_mse_loss(output, target):
    n, d = target.shape
    loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
    return loss

def logistic_loss(output, target):
    loss = 1.0 / target.shape[0] * (torch.logaddexp(torch.tensor(0.), output) - target * output).sum()
    return loss

def poisson_loss(output, target):
    S = torch.exp(output)
    loss = 1.0 / target.shape[0] * (S - output * target).sum()
    return loss