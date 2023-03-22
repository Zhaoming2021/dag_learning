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
""" 
n, d, s0, graph_type, sem_type = 1000, 20, 20, 'ER', 'mlp'
B_true = utils.simulate_dag(d, s0, graph_type)
X = utils.simulate_nonlinear_sem(B_true, n, sem_type)
#print(X)

#model = models.mlp_unsigned(dims=[d, 10, 1], bias=True)
model = models.mlp_signed(d, verbose=False, dtype=torch.double)
h = dag_function.notears(model, model_type='linear_unsigned') 
h_func = h.eval()
print(f'h_func',h_func)
X_torch = torch.from_numpy(X)
tstart = timer()
X_hat = model(X_torch)

loss_fn = scores.LossFunction()
score = loss_fn.log_mse_loss(X_hat, X_torch) 
print(score)
"""
n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
B_true = utils.simulate_dag(d, s0, graph_type)
W_true = utils.simulate_parameter(B_true)
np.savetxt('W_true.csv', W_true, delimiter=',')

X = utils.simulate_linear_sem(W_true, n, sem_type)
X_torch = torch.from_numpy(X)
np.savetxt('X.csv', X, delimiter=',')
model = models.linear_unsigned(d, verbose=False, dtype=torch.double)
h = dag_function.notears(model, model_type='linear_unsigned') 
h_func = h.eval()
print(f'h_func',h_func)
W = model.adj()
loss_fn = scores.LossFunction()
score = loss_fn.logistic_loss(X_torch,torch.matmul(X_torch,W))
print(score)
""" 
#torch.autograd.set_detect_anomaly(True)
W_est_ = algorithms_new.dagma_algo(model, h_func,score, model_type='mlp_signed', loss_type = None)
W_est = W_est_._algo(model, X_torch, lambda1=0.02, lambda2=0.005)
# assert utils.is_dag(W_est)
np.savetxt('W_est.csv', W_est, delimiter=',')
tend = timer()
acc = utils.count_accuracy(B_true, W_est != 0)
print(f'runtime: {tend-tstart}')
print(acc)

"""


