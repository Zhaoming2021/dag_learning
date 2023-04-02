# augmented_lagrangize.py
# penalty_method.py
# quadratic_penalty.py

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from  torch import optim
import copy
import scores as S
import models
import scipy.linalg as sla
import numpy.linalg as la
from lbfgsb_scipy import LBFGSBScipy


class AugmentedLagrangize:

    def __init__(self,model,loss,h_func,optimizers):
        self.model = model
        self.loss = loss
        self.h_func = h_func
        self.optimizers = optimizers

    def fit(self, X, lambda1,lambda2, w_threshold=0.3, max_iter=6e4, 
            checkpoint=1000, rho =1.0, alpha =0.0, h = np.inf, h_tol=1e-8, rho_max=1e+16
        ):
        """ the standard machinery of augmented Lagrangian, resulting in a series of unconstrained problems"""

        ## INITALIZING VARIABLES 
        self.X, self.lambda1,self.lambda2, self.checkpoint = X, lambda1,lambda2 ,checkpoint
        self.n, self.d = X.shape
        self.w_est = torch.zeros(2 * self.d * self.d, dtype=torch.float)  # double w_est into (w_pos, w_neg)
        #bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(self.d) for j in range(self.d)]

        for _ in range(max_iter):
            rho, alpha, h = self.dual_ascent_step(self.model, self.lambda1, self.lambda2,
                                         rho, alpha, h, rho_max)
             
            if h <= h_tol or rho >= rho_max:
                break
        W_est = self.model.adj(self.w_est)
        W_est[np.abs(W_est) < w_threshold] = 0
        return W_est
    
    def _minimize(self,lambda1,rho_max,rho,alpha,h):
        w_new, h_new = None, None
        optimizer = LBFGSBScipy(self.model.parameters())
        while rho < rho_max:
                def closure():
                    optimizer.zero_grad()
                    loss = self.loss
                    h_val = self.h_func
                    obj = loss + 0.5 * rho * h_val * h_val + alpha * h_val + lambda1 * w_new.sum()
                    obj.backward()
                    return obj               
                optimizer.step(closure)  # NOTE: updates model in-place
                with torch.no_grad():
                    h_new = self.h_func.item()
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
        self.w_est, h = w_new, h_new
        alpha += rho * h #Dual ascent
        return rho, alpha, h 

    def dual_ascent_step(self, lambda1, lambda2, rho, alpha, h, rho_max):
        """ single unconstrained problem """
        """Perform one step of dual ascent in augmented Lagrangian."""
        h_new = None
        optimizer = LBFGSBScipy(self.model.parameters())
        #X_torch = torch.from_numpy(X)
        while rho < rho_max:
            def closure():
                optimizer.zero_grad()
                #X_hat = model(X_torch)
                #loss = squared_loss(X_hat, X_torch)
                loss = self.loss
                #h_val = model.h_func()
                h_val = self.h_func
                penalty = 0.5 * rho * h_val * h_val + alpha * h_val
                l2_reg = 0.5 * lambda2 * self.model.l2_reg()
                l1_reg = lambda1 * self.model.fc1_l1_reg()
                primal_obj = loss + penalty + l2_reg + l1_reg
                primal_obj.backward()
                return primal_obj
            optimizer.step(closure)  # NOTE: updates model in-place
            with torch.no_grad():
                h_new = self.h_func.item()
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        alpha += rho * h_new
        h = h_new
        return rho, alpha, h
        

class PenaltyMethod:

    def __init__(self, model,loss,h_func,optimizers):
        self.model = model
        self.loss = loss
        self.h_func = h_func
        self.optimizers = optimizers

    def fit(self,X, lambda1, lambda2, beta1 = .99, beta2=.999, T=5, mu_init=.1, mu_factor=.1, s=1.0,
            warm_iter=3e4, max_iter=6e4, lr=0.0003, w_threshold=0.3, checkpoint=1000,verbose=False):
        
        ## INITALIZING VARIABLES 
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)


        self.cov = X.T @ X / float(self.n)    
        self.W_est = np.zeros((self.d,self.d)).astype(self.dtype) # init W0 at zero matrix

        vprint = print if verbose else lambda *a, **k: None
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
            self.vprint(f'\nIteration -- {i+1}:')
            lr_adam, success = lr, False
            inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
            while success is False:
                    W_temp, success = self.minimize(self.W_est.copy(), mu, inner_iters, s[i], lr=lr_adam, beta_1=beta_1, beta_2=beta_2, pbar=None)
                    if success is False:
                        self.vprint(f'Retrying with larger s')
                        lr_adam *= 0.5
                        s[i] += 0.1
            self.W_est = W_temp
            mu *= mu_factor

        ## Store final h and score values and threshold
        self.h_final= self.h_func(self.W_est)
        self.score_final = self.loss(self.W_est)
        self.W_est[np.abs(self.W_est) < w_threshold] = 0
        return self.W_est
    
    def mininize(self, W, mu, max_iter, s, lr, lr_decay=False, tol=1e-6, beta_1=0.99, beta_2=0.999, pbar=None):
 
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*beta_2)
        if lr_decay is True:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        self.vprint(f'\n\nMinimize with -- mu:{mu} -- lr: {lr} -- s: {s} -- l1: {self.lambda1} for {max_iter} max iterations')

        for iter in range(1, max_iter+1):
            optimizer.zero_grad()
            h_val = self.h_func()
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {iter}')
                return False
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < 0): # sI - W o W is not an M-matrix
                if iter == 1 or s <= 0.9:
                    self.vprint(f'W went out of domain for s={s} at iteration {iter}')
                    return W, False
            obj = mu*(self.loss +  self.beta_1 * np.sign(W)) + h_val
            obj.backward()
            optimizer.step()
            if lr_decay and (iter+1) % 1000 == 0: #every 1000 iters reduce lr
                scheduler.step()
            ## Check obj convergence
            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new = obj.item()
                self.vprint(f'\nInner iteration {iter}')
                self.vprint(f'\th(W(model)): {h_val.item()}')
                self.vprint(f'\tscore(model): {obj_new}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    break
                obj_prev = obj_new
        return W, True

                

                

                
