# augmented_lagrangize.py
# penalty_method.py
# quadratic_penalty.py
 
import numpy as np
from  torch import optim
import copy
import torch

class PenaltyMethod:

    def __init__(self, model, loss, h, optimizer):
        self.model = model
        self.loss = loss
        self.h = h
        self.optimizer = optimizer
    
    def fit(self, X: torch.tensor, lambda1= .02, lambda2 = .005,
        T=4, mu_init=.1, mu_factor=.1, s=[1.0, .9, .8, .7, .6], warm_iter=5e4, max_iter=8e4, lr=.0002, 
        w_threshold=0.3, checkpoint=1000, verbose=False):
        """ the standard machinery of augmented Lagrangian, resulting in a series of unconstrained problems""" 
        if type(X) is not torch.Tensor:
            ValueError("X should be tensor")
    
        ## INITALIZING VARIABLES 
        self.vprint = print if verbose else lambda *a, **k: None
        mu = mu_init
        if type(s) == list:
            if len(s) < T: 
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.") 

        for i in range(int(T)):
            self.vprint(f'\nDagma iter t={i+1} -- mu: {mu}', 30*'-')
            success, s_cur = False, s[i]
            inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
            model_copy = copy.deepcopy(self.model)
            lr_decay = False
            while success is False:
                success = self.minimize(X, lr, inner_iters, lambda1, lambda2, mu, s_cur, 
                                        lr_decay, checkpoint=checkpoint, verbose=verbose)
                if success is False:
                    self.model.load_state_dict(model_copy.state_dict().copy())
                    lr  *= 0.5
                    lr_decay = True
                    if lr < 1e-10:
                        break # lr is too small
                    s_cur += 0.1
            mu *= mu_factor   
        W_est = self.model.adj().detach().numpy()
        W_est[np.abs(W_est) < w_threshold] = 0
        return W_est
    
    def minimize(self, X, lr, max_iter, lambda1, lambda2, mu, s, lr_decay=False, checkpoint=1000, tol=1e-6, verbose=False):
        """ single unconstrained problem """
        self.vprint = print if verbose else lambda *a, **k: None
        self.vprint(f'\nMinimize s={s} -- lr={lr}')
        optimizer = self.optimizer
        if lr_decay is True:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        obj_prev = 1e16
        for i in range(max_iter):
            optimizer.zero_grad()
            h_val = self.h.eval(s)
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False 
            X_hat = self.model(X) 
            score = self.loss(X_hat, X)
            l1_reg = lambda1 * self.model.l1_loss()
            l2_reg = lambda2 * self.model.l2_loss()
            obj = mu * (score + l1_reg + l2_reg) + h_val
            obj.backward()
            optimizer.step()
            if lr_decay and (i+1) % 1000 == 0: #every 1000 iters reduce lr
                scheduler.step()
            if i % checkpoint == 0 or i == max_iter-1:
                obj_new = obj.item()
                self.vprint(f"\nInner iteration {i}")
                self.vprint(f'\th(W(model)): {h_val.item()}')
                self.vprint(f'\tscore(model): {obj_new}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    break
                obj_prev = obj_new
        return True

class Augmented_lagrangize:

    def __init__(self, model, loss, h, optimizer):
        self.model = model
        self.loss = loss
        self.h = h
        self.optimizer = optimizer

    def fit(self,X: np.ndarray,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
        ## INITALIZING VARIABLES 
        rho, alpha, h = 1.0, 0.0, np.inf
        for _ in range(max_iter):
            rho, alpha, h = self.dual_ascent_step(X, lambda1, lambda2,
                                         rho, alpha, h, rho_max)
            if h <= h_tol or rho >= rho_max:
                break
        W_est = self.model.adj().detach().numpy()
        W_est[np.abs(W_est) < w_threshold] = 0
        return W_est
    
    def dual_ascent_step(self, X, lambda1, lambda2, rho, alpha, h, rho_max):
        """Perform one step of dual ascent in augmented Lagrangian."""
        h_new = None
        optimizer = self.optimizer
        while rho < rho_max:
            def closure():
                optimizer.zero_grad()
                X_hat = self.model(X)
                loss = self.loss(X_hat, X)
                h_val = self.h.eval()
                penalty = 0.5 * rho * h_val * h_val + alpha * h_val
                l2_reg = 0.5 * lambda2 * self.model.l2_loss()
                l1_reg = lambda1 * self.model.l1_loss()
                primal_obj = loss + penalty + l2_reg + l1_reg
                primal_obj.backward()
                return primal_obj
            optimizer.step(closure)  # NOTE: updates model in-place
            with torch.no_grad():
                h_new = self.h.eval().item()
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        alpha += rho * h_new
        return rho, alpha, h_new


    