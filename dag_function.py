# notears.py
# dagma.py
# poly.py
import torch
import torch.nn as nn
import numpy as np
from trace_expm import trace_expm


class dagma:
  def __init__(self, model):
    self.model = model

  def eval(self, W, s=1.0,):
    if self.model.type == "mlp_signed":
      #similar to h in dagma nonlinear
      fc1_weight = self.model.fc1.weight
      fc1_weight = fc1_weight.view(self.model.d, -1, self.model.d)
      A = torch.sum(fc1_weight ** 2, dim=1).t()
      h = -torch.slogdet(s * self.model.I - A)[1] + self.model.d * np.log(s)
      return h
    if self.model.type == "linear_signed":
      # similar to h in dagma linear
      M = s * self.model.Id - W @ W
      h = - torch.slogdet(M)[1] + self.model.dims[0] * np.log(s)
      G_h = 2 * W @ torch.inverse(M).T 
      return h, G_h

class notears:
  def __init__(self, model):
    self.model = model

  def eval(self,W):
    if self.model.type == "mlp_unsigned":
       #similar to h in notears nonlinear
       fc1_pos_weight = self.model.fc1_pos.weight
       fc1_neg_weight = self.model.fc1_neg.weight
       fc1_weight = fc1_pos_weight - fc1_neg_weight
       fc1_weight = fc1_weight.view(self.model.d, -1, self.model.d)
       A =torch.sum(fc1_weight * fc1_weight, dim=1).t()
       h = trace_expm(A) - self.model.d
       return h
      
    if self.model.type == "linear_unsigned":
      #similar to h in notears linear
      E = torch.matrix_exp(W @ W)  # (Zheng et al. 2018)
      h = torch.trace(E) - self.model.d
      G_h = torch.matmul(torch.transpose(E, 0, 1), W) * 2
      return h, G_h


# how to use this class
""" 
import dag_function

h = dag_function.dagma(model) 
### now I can call eval to h anywhere I want like this, e.g., for s=0.9
h.eval(s=0.9)

### similarly for notears all I need to do is
h = dag_function.notears(model) 
h.eval() 
"""