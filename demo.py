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

dims=[d, 10, 1]
model = models.mlp_signed(dims=dims, bias=True)
h = dag_function.nonlinear(model, model_type='mlp_signed', dims=dims) 
h_func = h.eval(s=1.0)
#print(f'h_func',h_func)
X_torch = torch.from_numpy(X)
tstart = timer()
X_hat = model(X_torch)

loss_fn = scores.LossFunction()
score = loss_fn.log_mse_loss(X_hat, X_torch) 

W_est_ = algorithms_new.dagma_algo(model, h_func, score, model_type='mlp_signed', loss_type = None)
W_est = W_est_._algo(model, X_torch, lambda1=0.02, lambda2=0.005)
# assert utils.is_dag(W_est)
np.savetxt('W_est.csv', W_est, delimiter=',')
tend = timer()
acc = utils.count_accuracy(B_true, W_est != 0)
print(f'runtime: {tend-tstart}')
print(acc)

