import numpy as np
import torch
import torch.nn as nn
import models
import scores
import dag_function
import algorithms
from  torch import optim
from timeit import default_timer as timer
import utils

# nonlinear example

torch.set_default_dtype(torch.double)
utils.set_random_seed(1)
torch.manual_seed(1)
n, d, s0, graph_type, sem_type = 1000, 20, 20, 'ER', 'mlp'
B_true = utils.simulate_dag(d, s0, graph_type)
#W_true = utils.simulate_parameter(B_true)
#X = utils.simulate_nonlinear_sem(W_true, n, sem_type)  
X = utils.simulate_nonlinear_sem(B_true, n, sem_type)

# in the demo.py
model = models.mlp_signed(dims=[d, 10, 1], bias=True)
h_func = dag_function.dag_functions(model, X, dims = [d, 10, 1], s=1.0)
X_hat = model(X)
loss_fn = scores.LossFunction()
score = loss_fn.log_mse_loss(X_hat, X) 
# optimizer = optim.adam(model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*lambda2) # how to solve each unconstrained problem

X_torch = torch.from_numpy(X)
W_est = algorithms.dagma_nonlinear(model,score, h_func, X_torch, lambda1=0.02, lambda2=0.005)
#W_est  = algorithms.fit(model, score, h_func, optimizer)
assert utils.is_dag(W_est)
np.savetxt('W_est.csv', W_est, delimiter=',')
acc = utils.count_accuracy(B_true, W_est != 0)
print(acc)

