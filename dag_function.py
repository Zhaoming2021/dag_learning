import torch
import numpy as np

class dagma:
  
  def __init__(self, model):
    self.model = model
    
  def eval(self, s=1.0):
      W = self.model.adj()
      M = s * self.model.I - W*W
      h = - torch.slogdet(M)[1] + self.model.d * np.log(s)
      return h 

class notears:
   
  def __init__(self,model):
      self.model = model

  def eval(self):
      """Evaluate value and gradient of acyclicity constraint. """
      W = self.model.adj()
      E= torch.matrix_exp(W * W)  
      h = torch.trace(E) - self.model.d
      return h

#how to use this class 
""" 
import dag_function

h = dag_function.dagma(model) 
### now I can call eval to h anywhere I want like this, e.g., for s=0.9
h.eval(s=0.9)

### similarly for notears all I need to do is
h = dag_function.notears(model) 
h.eval(W) 
"""