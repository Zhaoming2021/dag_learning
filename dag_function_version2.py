import torch
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