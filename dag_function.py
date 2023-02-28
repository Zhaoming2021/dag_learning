# notears.py
# dagma.py
# poly.py
import torch
import torch.nn as nn
import numpy as np
import algorithms
import models
from tqdm.auto import tqdm
import scipy.optimize as sopt

# notears for linear
class dag_function:
    def __init__(self, model_type, loss_type='l2', dtype=np.float64):
        self.loss_type = loss_type
        self.model_type = model_type
        self.dtype = dtype

        if model_type == "notears_linear":
            
            self.dag_fn = self.notears_linear
        
        elif model_type == "dagma_linear":
            self.loss_fn = self.fit

        elif model_type == "dagma_mlp":
            self.loss_fn = self.dagma_nonlinear

        elif model_type == "notears_mlp":
            self.loss_fn = self.notears_nonlinear
            
        else:
            raise ValueError("Invalid model type")



    def notears_linear(X, lambda_1=0., lambda_2=0., loss_type='l2', h_tol=1e-4, 
                        w_threshold=1e-4, rho_max=1e+16, max_iter=1000):
        """
        Learn a linear DAG from observational data using the NOTEARS algorithm.

        Args:
            X (ndarray): n x d data matrix
            lambda_1 (float): l1 regularization parameter
            lambda_2 (float): l2 regularization parameter
            loss_type (str): 'l2' or 'log' for L2 loss or logistic loss, respectively
            h_tol (float): tolerance for constraint violation, the algorithm stops if vilation is below this level
            w_threshold (float): threshold for zeroing out small edges
            rho_max (float): maximum value for rho, the ADMM penalty parameter
            max_iter (int): maximum number of iterations to run the algorithm

        Returns:
            ndarray: d x d DAG adjacency matrix
        """
        n, d = X.shape
        w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
        bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
        if loss_type == 'l2':
            X = X - np.mean(X, axis=0, keepdims=True)
        for _ in range(max_iter):
            w_new, h_new = None, None
            while rho < rho_max:
                sol = sopt.minimize(models._func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
                w_new = sol.x
                h_new, _ = models._h(models._adj(w_new))
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            w_est, h = w_new, h_new
            alpha += rho * h
            if h <= h_tol or rho >= rho_max:
                break
        W_est = models._adj(w_est)
        W_est[np.abs(W_est) < w_threshold] = 0
        return W_est


    #notears for mlp

    def notears_nonlinear(model: nn.Module,
                        X: np.ndarray,
                        lambda1: float = 0.,
                        lambda2: float = 0.,
                        max_iter: int = 100,
                        h_tol: float = 1e-8,
                        rho_max: float = 1e+16,
                        w_threshold: float = 0.3):
        rho, alpha, h = 1.0, 0.0, np.inf
        for _ in range(max_iter):
            rho, alpha, h = algorithms.dual_ascent_step(model, X, lambda1, lambda2,
                                            rho, alpha, h, rho_max)
            if h <= h_tol or rho >= rho_max:
                break
        W_est = model.fc1_to_adj()
        W_est[np.abs(W_est) < w_threshold] = 0
        return W_est


    # dagma for linear
    def fit(self, X, lambda1, w_threshold=0.3, T=5,
                mu_init=1.0, mu_factor=0.1, s=[1.0, .9, .8, .7, .6], 
                warm_iter=3e4, max_iter=6e4, lr=0.0003, 
                checkpoint=1000, beta_1=0.99, beta_2=0.999,
            ):
            ## INITALIZING VARIABLES 
            self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
            self.n, self.d = X.shape
            self.Id = np.eye(self.d).astype(self.dtype)
            
            if self.loss_type == 'l2':
                self.X -= X.mean(axis=0, keepdims=True)
                
            self.cov = X.T @ X / float(self.n)    
            self.W_est = np.zeros((self.d,self.d)).astype(self.dtype) # init W0 at zero matrix
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
            with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:
                for i in range(int(T)):
                    self.vprint(f'\nIteration -- {i+1}:')
                    lr_adam, success = lr, False
                    inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                    while success is False:
                        W_temp, success = self.minimize(self.W_est.copy(), mu, inner_iters, s[i], lr=lr_adam, beta_1=beta_1, beta_2=beta_2, pbar=pbar)
                        if success is False:
                            self.vprint(f'Retrying with larger s')
                            lr_adam *= 0.5
                            s[i] += 0.1
                    self.W_est = W_temp
                    mu *= mu_factor
            
            ## Store final h and score values and threshold
            self.h_final, _ = self._h(self.W_est)
            self.score_final, _ = self._score(self.W_est)
            self.W_est[np.abs(self.W_est) < w_threshold] = 0
            return self.W_est


    # dagma for mlp

    def dagma_nonlinear(
        model: nn.Module, X: torch.tensor, lambda1=.02, lambda2=.005,
        T=4, mu_init=.1, mu_factor=.1, s=1.0,
        warm_iter=5e4, max_iter=8e4, lr=.0002, 
        w_threshold=0.3, checkpoint=1000, verbose=False
    ):
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
                    success = algorithms.minimize(model, X, inner_iter, lr, lambda1, lambda2, mu, s_cur, 
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