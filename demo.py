import torch
import torch.nn as nn
import numpy as np
import scipy.linalg as slin
import scipy.linalg as sla
import numpy.linalg as la
import torch.nn.functional as F
from locally_connected import LocallyConnected
from trace_expm import trace_expm
import dag_function
import models


if __name__ == '__main__':
    from timeit import default_timer as timer
    import utils
    
    torch.set_default_dtype(torch.double)
    utils.set_random_seed(1)
    torch.manual_seed(1)
    
    n, d, s0, graph_type, sem_type = 1000, 20, 20, 'ER', 'mlp'
    B_true = utils.simulate_dag(d, s0, graph_type)
    X = utils.simulate_nonlinear_sem(B_true, n, sem_type)

    model = models.dagma_mlp(dims=[d, 10, 1], bias=True)
    X_torch = torch.from_numpy(X)
    tstart = timer()
    W_est = dag_function.dagma_nonlinear(model, X_torch, lambda1=0.02, lambda2=0.005)
    tend = timer()
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(f'runtime: {tend-tstart}')
    print(acc)





# in the demo.py
model = mlp_signed(hyperparams)
h_func = notears(model,hyperparams)
score = mse_loss(data, model,hyperparams) 
optimizer = adam(hyperparams) # how to solve each unconstrained problem

W_est  = fit.augmented_lagrangian(model, score, h_func, optimizer)