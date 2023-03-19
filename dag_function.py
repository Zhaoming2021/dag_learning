import torch
import numpy as np
import torch.nn as nn


class nonlinear:
  def __init__(self,model, model_type, dims):
    self.model = model
    self.model_type = model_type
    self.dims, self.d = dims, dims[0]
    self.I = torch.eye(self.d)
    #self.fc1 = nn.Linear(self.d, self.d * dims[1], bias=bias)
    #self.fc1_pos = nn.Linear(self.d, self.d * dims[1], bias=bias)
    #self.fc1_neg = nn.Linear(self.d, self.d * dims[1], bias=bias)
    self.k = dims[-1]

  def eval(self, s=1.0):
    if self.model_type == "mlp_signed":
      fc1_weight = self.model.fc1.weight # not sure can call fc1.weight in this way
      fc1_weight = fc1_weight.view(self.d, -1, self.d)
      A = torch.sum(fc1_weight ** 2, dim=1).t()  # [i, j]
      h = -torch.slogdet(s * self.I - A)[1] + self.d * np.log(s)
      #h = -torch.slogdet(s * torch.eye(self.model.d) - A)[1] + self.model.d * torch.log(s)
      return h

    if self.model_type == "mlp_unsigned":
      fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
      fc1_weight = fc1_weight.view(self.d, -1, self.d)  # [j, i, k]
      #fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
      A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
      #A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
      h = torch.trace(torch.matrix_exp(A)) - self.d
      return h 
    
class linear:
  def __init__(self,model, model_type):
    self.model = model
    self.model_type = model_type

  def eval(self,W,s=1.0):
    if self.model_type == "linear_signed":
      M = s * torch.eye(self.model.d) - torch.matmul(W, W)    # in numpy : s * self.Id - W * W
      h = - torch.slogdet(M)[1] + self.model.d * torch.log(s)
      G_h = 2 * torch.matmul(W, torch.inverse(M).T)
      return h, G_h
    
    if self.model_type == "linear_unsigned":
      E = torch.matrix_exp(W * W)  
      h = torch.trace(E) - self.model.d
      G_h = torch.matmul(torch.matmul(E.t(), W), 2)
      return h, G_h 
  



# how to use this class
""" 
import dag_function

h = dag_function.nonlinear(model) 
### now I can call eval to h anywhere I want like this, e.g., for s=0.9
h.eval(s=0.9)

### similarly for notears all I need to do is
h = dag_function.linear(model) 
h.eval(W,s=0.9) 
"""


