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

import torch
import numpy as np
import scipy.linalg as slin

class notears:
  def __init__(self, model):
    self.model = model

  def _h(self, W):
    """Evaluate value and gradient of acyclicity constraint."""
    E = slin.expm(W * W)  
    h = np.trace(E) - self.d
    G_h = E.T * W * 2
    return h, G_h

  def h_func(self):
    if self.model.type == "mlp_unsigned":
      fc1_weight = self.model.fc1_pos.weight - self.model.fc1_neg.weight
      fc1_weight = fc1_weight.view(self.d, self.d, self.k)
      A = torch.sum(fc1_weight * fc1_weight, dim=2).t()
      h = self._h(A)[0]
    elif self.model.type == "linear_unsigned":
      A = self.model.weight.t()
      h = self._h(A)[0]
    else:
      raise ValueError("Unknown model type")
    return h

  def eval(self, ...):
    h = self.h_func()
    # rest of the evaluation code here



  """ def eval(self,W):
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
      return h, G_h """


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








import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class notears:
    
    def __init__(self, model_type, d, k):
        self.type = model_type
        self.d = d
        self.k = k
        if self.type == "mlp_unsigned":
            self.fc1 = weight_norm(torch.nn.Linear(self.d, self.d * self.k))
            self.fc2 = torch.nn.Linear(self.d * self.k, self.d)
        elif self.type == "linear_unsigned":
            self.fc1_pos = weight_norm(torch.nn.Linear(self.d, self.d * self.k, bias=False))
            self.fc1_neg = weight_norm(torch.nn.Linear(self.d, self.d * self.k, bias=False))
    
    def eval(self, W): 
        if self.type == "mlp_unsigned":
            fc1_weight = self.fc1.weight.view(self.d, self.k, self.d)  # [i, k, j]
            fc2_weight = self.fc2.weight  # [j, i*k]
            x = F.relu(self.fc1(W))
            x = self.fc2(x.view(-1, self.d * self.k))
            A = torch.mm(fc1_weight.view(-1, self.d), fc2_weight.t())  # [i*k, j]
            h = torch.trace(torch.matrix_exp(torch.mm(A.t(), A))) - self.d
            G_h = 2 * torch.mm(torch.mm(A, x.unsqueeze(-1)), fc1_weight.permute(2, 0, 1)).squeeze(-1)
        elif self.type == "linear_unsigned":
            fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
            fc1_weight = fc1_weight.view(self.d, self.d, self.k)  # [j, i, k]
            A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
            h = torch.trace(torch.matrix_exp(A.t() @ A)) - self.d  
            G_h = 2 * (A @ W @ fc1_weight.permute(1, 0, 2)).permute(1, 0, 2)
        
        return h, G_h





############################################# new #################################
import torch
from trace_expm import trace_expm

class dagma:
    
    def __init__(self,model):
        self.model = model

    def h_func(self, s=1.0):
      """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG,"mlp_signed":"""
      fc1_weight = self.model.fc1.weight
      fc1_weight = fc1_weight.view(self.model.d, -1, self.model.d)
      A = torch.sum(fc1_weight ** 2, dim=1).t()  # [i, j]
      h = -torch.slogdet(s * torch.eye(self.model.d) - A)[1] + self.model.d * torch.log(s)
      return h
    
    def _h(self, W, s=1.0):
        M = s * torch.eye(self.model.d) - torch.matmul(W, W)    # in numpy : s * self.Id - W * W
        h = - torch.slogdet(M)[1] + self.model.d * torch.log(s)
        G_h = 2 * torch.matmul(W, torch.inverse(M).T)
        return h, G_h
    
    def eval(self, W, s=1.0):
        if self.model.type == "mlp_signed":
            return self.h_func(s)
        elif self.model.type == "linear_signed":
            return self._h(W, s)
        else:
            raise ValueError("Invalid model type.")
        
class notears:
   
    def __init__(self,model):
        self.model = model

    def _h(self, W):
        """Evaluate value and gradient of acyclicity constraint. """
        E = torch.matrix_exp(W * W)  
        h = torch.trace(E) - self.model.d
        G_h = torch.matmul(torch.matmul(E.t(), W), 2)
        return h, G_h   
    
    def h_func(self):
        fc1_weight = self.model.fc1_pos.weight - self.model.fc1_neg.weight  # [j, ik]
        fc1_weight = fc1_weight.view(self.model.d, self.model.d, self.model.k)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        h = torch.trace(torch.matrix_exp(A)) - self.model.d
        return h 
    
    def eval(self, W):
        if self.model.type == "mlp_unsigned":
            return self.h_func()
        elif self.model.type == "linear_unsigned":
            return self._h(W)
        else:
            raise ValueError("Invalid model type.")
        


# how to use this class
""" 
import dag_function

h = dag_function.dagma(model) 
### now I can call eval to h anywhere I want like this, e.g., for s=0.9
h.eval(W,s=0.9)

### similarly for notears all I need to do is
h = dag_function.notears(model) 
h.eval(W) 
"""