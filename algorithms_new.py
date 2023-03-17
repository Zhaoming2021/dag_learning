import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from  torch import optim
import copy
from scipy.special import expit as sigmoid
import scipy.linalg as sla
import scipy.optimize as sopt
from lbfgsb_scipy import LBFGSBScipy


class dagma_algo:
    def __init__(self, model, h_func, score, model_type, loss_type):
        self.model = model
        self.h_func = h_func
        self.score = score
        self.model_type = model_type
        self.loss_type = loss_type

    def minimize(self, model, X, max_iter, lr, lambda1, lambda2, mu, s, lr_decay=False, checkpoint=1000, tol=1e-6, verbose=False, pbar=None):
        vprint = print if verbose else lambda *a, **k: None
        vprint(f'\nMinimize s={s} -- lr={lr}')
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*lambda2)
        if lr_decay is True:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        obj_prev = 1e16
        for i in range(max_iter):
            optimizer.zero_grad()
            h_val = self.h_func
            #h_val = dag_function.h(model, X, s=1.0)
            if h_val.item() < 0:
                vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            #X_hat = model(X)
            #score = scores.log_mse_loss(X_hat, X)
            score = self.score
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
    
    def minimize_unsigned(self, W, mu, max_iter, s, lr, tol=1e-6, beta_1=0.99, beta_2=0.999, pbar=None):
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        self.vprint(f'\n\nMinimize with -- mu:{mu} -- lr: {lr} -- s: {s} -- l1: {self.lambda1} for {max_iter} max iterations')
        
        for iter in range(1, max_iter+1):
            ## Compute the (sub)gradient of the objective
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < 0): # sI - W o W is not an M-matrix
                if iter == 1 or s <= 0.9:
                    self.vprint(f'W went out of domain for s={s} at iteration {iter}')
                    return W, False
                else:
                    W += lr * grad
                    lr *= .5
                    if lr <= 1e-16:
                        return W, True
                    W -= lr * grad
                    M = sla.inv(s * self.Id - W * W) + 1e-16
                    self.vprint(f'Learning rate decreased to lr: {lr}')
            
            if self.loss_type == 'l2':
                G_score = -mu * self.cov @ (self.Id - W) 
            elif self.loss_type == 'logistic':
                G_score = mu / self.n * self.X.T @ sigmoid(self.X @ W) - mu * self.cov
            Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T
            
            ## Adam step
            grad = self._adam_update(Gobj, iter, beta_1, beta_2)
            W -= lr * grad
            
            ## Check obj convergence
            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new, score, h = self._func(W, mu, s)
                self.vprint(f'\nInner iteration {iter}')
                self.vprint(f'\th(W_est): {h:.4e}')
                self.vprint(f'\tscore(W_est): {score:.4e}')
                self.vprint(f'\tobj(W_est): {obj_new:.4e}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter-iter+1)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return W, True


    def _algo(self, model: nn.Module, X: torch.tensor, lambda1=.02, lambda2=.005, beta_1=0.99, beta_2=0.999,
        T=4, mu_init=.1, mu_factor=.1, s=1.0, warm_iter=5e4, max_iter=8e4, lr=.0002, w_threshold=0.3, checkpoint=1000, verbose=False):

        if self.model_type == "mlp_signed":
            
            vprint = print if verbose else lambda *a, **k: None
            mu = mu_init
            if type(s) == list:
                if len(s) < T: 
                    vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                    s = s + (T - len(s)) * [s[-1]]
            elif type(s) in [int, float]:
                s = T * [s]
            else:
                ValueError("s should be a list, int, or float.") 
            with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:
                for i in range(int(T)):
                    vprint(f'\nDagma iter t={i+1} -- mu: {mu}', 30*'-')
                    success, s_cur = False, s[i]
                    inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
                    model_copy = copy.deepcopy(model)
                    lr_decay = False
                    while success is False:
                        success = self.minimize(model, X, inner_iter, lr, lambda1, lambda2, mu, s_cur, 
                                            lr_decay, checkpoint=checkpoint, verbose=verbose, pbar=pbar)
                        if success is False:
                            model.load_state_dict(model_copy.state_dict().copy())
                            lr *= 0.5 
                            lr_decay = True
                            if lr < 1e-10:
                                break # lr is too small
                            s_cur = 1
                    mu *= mu_factor
            W_est = model.fc1_to_adj()
            W_est[np.abs(W_est) < w_threshold] = 0
            return W_est

        if self.model_type == "linear_signed":
            X, lambda1, checkpoint = X, lambda1, checkpoint
            n, d = X.shape
            self.Id = np.eye(d).astype(np.float64)
            vprint = print if verbose else lambda *a, **k: None
            
            if self.loss_type == 'l2':
                X -= X.mean(axis=0, keepdims=True)
                
            self.cov = X.T @ X / float(n)    
            W_est = np.zeros((d,d)).astype(np.float64) # init W0 at zero matrix
            mu = mu_init
            if type(s) == list:
                if len(s) < T: 
                    vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                    s = s + (T - len(s)) * [s[-1]]
            elif type(s) in [int, float]:
                s = T * [s]
            else:
                ValueError("s should be a list, int, or float.")    
            
            ## START DAGMA
            with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:
                for i in range(int(T)):
                    vprint(f'\nIteration -- {i+1}:')
                    lr_adam, success = lr, False
                    inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                    while success is False:
                        W_temp, success = self.minimize_unsigned(W_est.copy(), mu, inner_iters, s[i], lr=lr_adam, beta_1=beta_1, beta_2=beta_2, pbar=pbar)
                        if success is False:
                            vprint(f'Retrying with larger s')
                            lr_adam *= 0.5
                            s[i] += 0.1
                    W_est = W_temp
                    mu *= mu_factor
            
            ## Store final h and score values and threshold
            h_final, _ = self.h_func(W_est)  #model, X, W, dims, s=1.0
            score_final, _ = self.score(W_est)  #?score 能否这么写
            W_est[np.abs(W_est) < w_threshold] = 0
            return W_est
            
        


class notears_algo:
    def __init__(self, model,h_func,score, model_type, loss_type):
        self.model = model
        self.h_func = h_func
        self.score = score
        self.model_type = model_type
        self.loss_type = loss_type

    def dual_ascent_step(self, model, X, lambda1, lambda2, rho, alpha, h, rho_max):
        """Perform one step of dual ascent in augmented Lagrangian."""
        h_new = None
        optimizer = LBFGSBScipy(model.parameters())
        #X_torch = torch.from_numpy(X)
        while rho < rho_max:
            def closure():
                optimizer.zero_grad()
                #X_hat = model(X_torch)
                #loss = squared_loss(X_hat, X_torch)
                loss = self.score
                #h_val = model.h_func()
                h_val = self.h_func
                penalty = 0.5 * rho * h_val * h_val + alpha * h_val
                l2_reg = 0.5 * lambda2 * model.l2_reg()
                l1_reg = lambda1 * model.fc1_l1_reg()
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
        return rho, alpha, h_new


    def _algo(self, model, X: np.ndarray,lambda1: float = 0.,lambda2: float = 0.,loss_type='l2', max_iter: int = 100,h_tol: float = 1e-8,rho_max: float = 1e+16,
w_threshold: float = 0.3):
        if self.model_type == "mlp_unsigned":
            rho, alpha, h = 1.0, 0.0, np.inf
            for _ in range(max_iter):
                rho, alpha, h = self.dual_ascent_step(model, X, lambda1, lambda2,
                                                rho, alpha, h, rho_max)
                if h <= h_tol or rho >= rho_max:
                    break
            W_est = self.model.fc1_to_adj()
            W_est[np.abs(W_est) < w_threshold] = 0
            return W_est


        if self.model_type == "linear_unsigned":
            n, d = X.shape
            w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
            bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
            if loss_type == 'l2':
                X = X - np.mean(X, axis=0, keepdims=True)
            for _ in range(max_iter):
                w_new, h_new = None, None
                while rho < rho_max:
                    sol = sopt.minimize(model._func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
                    w_new = sol.x
                    h_new, _ = eval(self.model._adj(w_new))
                    if h_new > 0.25 * h:
                        rho *= 10
                    else:
                        break
                w_est, h = w_new, h_new
                alpha += rho * h
                if h <= h_tol or rho >= rho_max:
                    break
            W_est = model._adj(w_est)
            W_est[np.abs(W_est) < w_threshold] = 0
            return W_est