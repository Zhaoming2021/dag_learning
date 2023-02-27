import scores
import models
import torch
import torch.nn as nn
import numpy as np
from lbfgsb_scipy import LBFGSBScipy
from  torch import optim
# linear for notears and dagma
class Algorithms:
     def __init__(self, model_type, loss_type=None):
        self.model_type = model_type
        self.loss_type = loss_type

        if model_type == "notears_linear":

            self._algo = self._func_notears()
            
        elif model_type == "dagma_linear":  

            self._algo = self._func_dag()
    
        elif model_type == "notears_mlp":
            self._algo = self.dual_ascent_step()

        elif model_type == "dagma_mlp":
            self._algo = self.minimize()
        
        else:
            raise ValueError("Invalid model type")
            
#linear for notears
def _func_notears(w,alpha,lambda1,rho):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = models._adj(w)
        loss, G_loss = scores._loss(W)
        h, G_h = models._h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

#linear for dagma
def _func_dag(W, mu,lambda1, s=1.0):
        """Evaluate value of the penalized objective function."""
        score, _ = scores._score(W)
        h, _ = models._h(W, s)
        obj = mu * (score + lambda1 * np.abs(W).sum()) + h 
        return obj, score, h

# mlp for notears
def dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    X_torch = torch.from_numpy(X)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            loss = scores.squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj
        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new

# mlp for dagma
def minimize(model, X, max_iter, lr, lambda1, lambda2, mu, s, lr_decay=False, checkpoint=1000, tol=1e-6, verbose=False, pbar=None):
    vprint = print if verbose else lambda *a, **k: None
    vprint(f'\nMinimize s={s} -- lr={lr}')
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*lambda2)
    if lr_decay is True:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    obj_prev = 1e16
    for i in range(max_iter):
        optimizer.zero_grad()
        h_val = model.h_func(s)
        if h_val.item() < 0:
            vprint(f'Found h negative {h_val.item()} at iter {i}')
            return False
        X_hat = model(X)
        score = score.log_mse_loss(X_hat, X)
        l1_reg = lambda1 * model.fc1_l1_reg()
        obj = mu * (score + l1_reg) + h_val
        obj.backward()
        optimizer.step()
        if lr_decay and (i+1) % 1000 == 0: #every 1000 iters reduce lr
            scheduler.step()
        if i % checkpoint == 0 or i == max_iter-1:
            obj_new = obj.item()
            vprint(f"\nInner iteration {i}")
            vprint(f'\th(W(model)): {h_val.item()}')
            vprint(f'\tscore(model): {obj_new}')
            if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                pbar.update(max_iter-i)
                break
            obj_prev = obj_new
        pbar.update(1)
    return True
