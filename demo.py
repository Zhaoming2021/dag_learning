import numpy as np
import torch
import models
import scores
import dag_function
import algorithms
import utils
from  torch import optim

### in the demo.py
## nonlinear example

torch.set_default_dtype(torch.double)
utils.set_random_seed(123)
torch.manual_seed(1)

# mlp example
n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'mlp'
B_true = utils.simulate_dag(d, s0, graph_type)
X = utils.simulate_nonlinear_sem(B_true, n, sem_type)
model = models.mlp_signed(dims=[d, 10, 1], bias=True, dtype=torch.double) #try linear 
h = dag_function.dagma(model)
X_torch = torch.from_numpy(X)
loss = scores.log_mse_loss
optimizer = optim.Adam(model.parameters(), betas=(.99,.999))
algo = algorithms.PenaltyMethod(model, loss, h, optimizer)
W_est = algo.fit(X_torch, lambda1=0.002, lambda2=0.005)
np.savetxt('W_est.csv', W_est, delimiter=',')
acc = utils.count_accuracy(B_true, W_est != 0)
print(acc)  
""" 
# linear example singed example
n, d, s0, graph_type, sem_type = 500, 20, 20, 'ER', 'gauss'
B_true = utils.simulate_dag(d, s0, graph_type)
W_true = utils.simulate_parameter(B_true)
np.savetxt('W_true.csv', W_true, delimiter=',')
X = utils.simulate_linear_sem(W_true, n, sem_type)
X_torch = torch.from_numpy(X)
model = models.linear_signed(d, verbose=False, dtype=torch.double)
h = dag_function.dagma(model) 
loss= scores.squared_loss
optimizer = optim.Adam(model.parameters(), betas=(.99,.999))
algo = algorithms.PenaltyMethod(model, loss, h, optimizer)
W_est = algo.fit(X_torch, lambda1=0.02)
#np.savetxt('W_est.csv', W_est, delimiter=',')
acc = utils.count_accuracy(B_true, W_est != 0)
print(acc) 
"""