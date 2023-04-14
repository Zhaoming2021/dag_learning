import numpy as np
import torch
import models
import scores
import dag_function
import algorithms
import utils
from  torch import optim

### in the demo.py


""" ## nonlinear example

torch.set_default_dtype(torch.double)
utils.set_random_seed(1)
torch.manual_seed(1)

# mlp example
n, d, s0, graph_type, sem_type = 10, 20, 20, 'ER', 'mlp'
B_true = utils.simulate_dag(d, s0, graph_type)
X = utils.simulate_nonlinear_sem(B_true, n, sem_type)
np.savetxt('X.csv', X, delimiter=',')

#model
#model = models.mlp_unsigned(dims=[d, 10, 1], bias=True, dtype=torch.float6)
model = models.mlp_signed(dims=[d, 10, 1], bias=True, dtype=torch.float64) #try linear 
# dag function
h = dag_function.dagma(model)
X_torch = torch.from_numpy(X)
#loss
loss = scores.log_mse_loss
#optimizers 
optimizer = optim.Adam(model.parameters(), betas=(.99,.999))
#algorithm
algo = algorithms.PenaltyMethod(model, loss, h, optimizer)
W_est = algo.fit(X_torch, lambda1=0.002, lambda2=0.005)
np.savetxt('W_est.csv', W_est, delimiter=',')
acc = utils.count_accuracy(B_true, W_est != 0)
print(acc) 
 """


# linear example
n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
B_true = utils.simulate_dag(d, s0, graph_type)
W_true = utils.simulate_parameter(B_true)
np.savetxt('W_true.csv', W_true, delimiter=',')

X = utils.simulate_linear_sem(W_true, n, sem_type)
X_torch = torch.from_numpy(X)
np.savetxt('X.csv', X, delimiter=',')
model = models.linear_signed(d, verbose=False, dtype=torch.double)
#print(list(model.__dict__.items()))
#print(type(model.__dict__.items()))
h = dag_function.dagma(model) 
loss= scores.logistic_loss
# optimizers
#optimizer = LBFGSBScipy(model.parameters())

params = [v for k, v in model.__dict__.items() if isinstance(v, torch.Tensor)]
params_tensor = torch.cat(params, dim=0)
params_list = torch.split(params_tensor, split_size_or_sections=len(params))
params_list_detached = tuple(p.detach() for p in params_list)
optimizer = optim.Adam(params_list_detached, betas=(.99, .999)) # don't need to have weight decay here just add l2 loss in the objective


algo = algorithms.PenaltyMethod(model, loss, h, optimizer)
W_est = algo.fit(X_torch, lambda1=0.02)
acc = utils.count_accuracy(B_true, W_est != 0)
print(acc) 


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


