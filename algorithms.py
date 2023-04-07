# augmented_lagrangize.py
# penalty_method.py
# quadratic_penalty.py


import numpy as np
from  torch import optim
import scipy.linalg as sla
import copy
import torch

class PenaltyMethod:

    def __init__(self, model, loss, h_func, optimizers):
        self.model = model
        self.loss = loss
        self.h_func = h_func
        self.optimizers = optimizers
    
    def fit(self, model, X, lambda1= .02, lambda2 = .005,
        T=4, mu_init=.1, mu_factor=.1, s=[1.0, .9, .8, .7, .6], warm_iter=5e4, max_iter=8e4, lr=.0002, 
        w_threshold=0.3, checkpoint=1000, verbose=False):
        """ the standard machinery of augmented Lagrangian, resulting in a series of unconstrained problems"""

        ## INITALIZING VARIABLES 
        self.vprint = print if verbose else lambda *a, **k: None
        self.n, self.d = X.shape
        self.Id = torch.eye(self.d, dtype=torch.float)
        mu = mu_init

        if type(s) == list:
            if len(s) < T: 
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.") 

        ## START DAGMA
        for i in range(int(T)):
            self.vprint(f'\nDagma iter t={i+1} -- mu: {mu}', 30*'-')
            lr_adam, success = lr, False
            s_cur = s[i]
            inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
            model_copy = copy.deepcopy(model)
            lr_decay = False

            if self.model.type == "linear_signed":
                self.cov = X.T @ X / float(self.n)    
                #self.W_est = np.zeros((self.d,self.d)).astype(self.dtype)
                W_est = torch.zeros((self.d,self.d))
                while success is False:
                    W_temp, success = self.minimize(W_est.copy(), mu, inner_iters, s_cur, lr=lr_adam)
                    if success is False:
                        self.vprint(f'Retrying with larger s')
                        lr_adam  *= 0.5
                        s_cur += 0.1
                W_est = W_temp
                mu *= mu_factor
            
            elif self.model.type == "mlp_signed":
                
                while success is False:
                    _, success = self.minimize(model, X, inner_iters, lambda1, lambda2, mu, s_cur, 
                                              lr_decay,lr = lr_adam, checkpoint=checkpoint, verbose=verbose)
                    if success is False:
                        model.load_state_dict(model_copy.state_dict().copy())
                        lr_adam  *= 0.5 
                        lr_decay = True
                        if lr_adam  < 1e-10:
                            break # lr is too small
                        s_cur = 1
                    mu *= mu_factor
            W_est = model.adj()
            
       
        W_est[np.abs(W_est) < w_threshold] = 0
        return W_est 


    def minimize(self, W, max_iter, lr, lambda1, lambda2, mu, s, lr_decay=False, checkpoint=1000, tol=1e-6, verbose=False):
        """ single unconstrained problem """
        self.vprint = print if verbose else lambda *a, **k: None
        self.vprint(f'\n\nMinimize with -- mu:{mu} -- lr: {lr} -- s: {s} -- l1: {self.lambda1} for {max_iter} max iterations')

        if lr_decay is True:
            scheduler = optim.lr_scheduler.ExponentialLR(self.optimizers, gamma=0.8)
        obj_prev = 1e16

        for i in range(max_iter):
            self.optimizers.zero_grad()
            h_val = self.h_func()
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            score = self.loss

            if self.model.type == "mlp_signed":
                l1_reg = lambda1 * self.model.fc1_l1_reg() 
                l2_reg = lambda2 * self.model.fc1_l2_reg()
                obj = mu * (score + l1_reg + l2_reg) + h_val
            elif self.model.type == "linear_signed":
                M = sla.inv(s * self.Id - W * W) + 1e-16
                while np.any(M < 0): # sI - W o W is not an M-matrix
                    if iter == 1 or s <= 0.9:
                        self.vprint(f'W went out of domain for s={s} at iteration {i}')
                        return W, False
                l1_norm = sum([p.abs().sum() for p in self.model.parameters()])
                l2_norm = sum([(p**2).sum() for p in self.model.parameters()])
                obj = mu*(score + lambda1*l1_norm + lambda2*l2_norm) + h_val    
            obj.backward()
            self.optimizers.step()
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