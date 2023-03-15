import numpy as np
import torch
import torch.nn as nn
import models
import scores
import dag_function
import algorithms_new
from timeit import default_timer as timer
import utils

### in the demo.py
## nonlinear example

torch.set_default_dtype(torch.double)
utils.set_random_seed(1)
torch.manual_seed(1)
n, d, s0, graph_type, sem_type = 1000, 20, 20, 'ER', 'mlp'
B_true = utils.simulate_dag(d, s0, graph_type)
X = utils.simulate_nonlinear_sem(B_true, n, sem_type)
#print(X)

model = models.mlp_signed(dims=[d, 10, 1], bias=True)
#W = model.fc1_to_adj()
#print(W)
h = dag_function.dagma(model) 
h_func = h.eval(W = None,s=1.0)
print(f'h_func',h_func)
X_torch = torch.from_numpy(X)
tstart = timer()
X_hat = model(X_torch)

loss_fn = scores.LossFunction()
score = loss_fn.log_mse_loss(X_hat, X_torch) 

W_est = algorithms_new.dagma_algo(model,h_func,score,loss_type = None)
W_est._algo(model, X_torch, lambda1=0.02, lambda2=0.005)
# assert utils.is_dag(W_est)
np.savetxt('W_est.csv', W_est, delimiter=',')
tend = timer()
acc = utils.count_accuracy(B_true, W_est != 0)
print(f'runtime: {tend-tstart}')
print(acc)

