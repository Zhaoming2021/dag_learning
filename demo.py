import numpy as np
import torch
import torch.nn as nn
import models
import scores
import dag_function
import algorithms
import utils
from lbfgsb_scipy import LBFGSBScipy
from  torch import optim

### in the demo.py
## nonlinear example

torch.set_default_dtype(torch.double)
utils.set_random_seed(1)
torch.manual_seed(1)


# mlp example
n, d, s0, graph_type, sem_type = 1000, 20, 20, 'ER', 'mlp'
B_true = utils.simulate_dag(d, s0, graph_type)
X = utils.simulate_nonlinear_sem(B_true, n, sem_type)

#model
#model = models.mlp_unsigned(dims=[d, 10, 1], bias=True, dtype=torch.float6)
model = models.mlp_signed(dims=[d, 10, 1], bias=True, dtype=torch.float64)

# dag function
h = dag_function.dagma(model) 
h_func = h.eval(s=1.0)
print(f'h_func',h_func)

X_torch = torch.from_numpy(X)
X_hat = model(X_torch)
#loss
loss = scores.log_mse_loss(X_hat, X_torch) 
print(loss)
#optimizers 
optimizers = optim.Adam(model.parameters(), betas=(.99,.999))

#algorithm
algo = algorithms.PenaltyMethod(model, loss, h_func, optimizers)
W_est = algo.fit(model, X_torch, lambda1=0.02, lambda2=0.005)
acc = utils.count_accuracy(B_true, W_est != 0)
print(acc)

""" 
# notears example
n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
B_true = utils.simulate_dag(d, s0, graph_type)
W_true = utils.simulate_parameter(B_true)
np.savetxt('W_true.csv', W_true, delimiter=',')

X = utils.simulate_linear_sem(W_true, n, sem_type)
X_torch = torch.from_numpy(X)
np.savetxt('X.csv', X, delimiter=',')
model = models.linear_unsigned(d, verbose=False, dtype=torch.double)
h = dag_function.notears(model) 
h_func = h.eval()
print(f'h_func',h_func)
W = model.adj()
score = scores.logistic_loss(X_torch,torch.matmul(X_torch,W))
print(score)
# optimizers
optimizer = LBFGSBScipy(model.parameters())
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*beta_2) # you don't need to have weight decay here just add l2 loss in the objective

algo = algorithms_1.PenaltyMethod(model,score,h_func,optimizer)
W_est = algo.fit(X_torch, lambda1=0.02)
acc = utils.count_accuracy(B_true, W_est != 0)
print(acc)

"""



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


