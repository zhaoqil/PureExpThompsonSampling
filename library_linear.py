import matplotlib.pyplot as plt
import numpy as np
#from IPython.display import clear_output
import multiprocessing as mp
from bandit_type import Linear
from utils import *
import utils
from distribution import *
import wandb

class ThompsonSampling(Linear):

    def __init__(self, X, Y, theta_star, T, sigma, name):
        super().__init__(X, Y, theta_star, T, sigma, name)
        self.B = np.matmul(X.reshape(-1,self.d,1), X.reshape(-1,1,self.d))
        self.Vinv = np.linalg.inv(self.V)
        self.pulled = np.zeros(X.shape[0])
        self._best_idx = np.argmax(X@theta_star)

    def run(self, logging_period=100):
        theta = np.zeros(self.d)
        S = 0
        for t in range(self.T):
            theta_hat = np.random.multivariate_normal(theta, self.Vinv)
            best_idx = np.argmax(self.X @ theta_hat)
            
            x_n = self.X[best_idx]
            y_n = x_n @ self.theta_star + self.sigma*np.random.randn()
            self.V += np.outer(x_n, x_n)
            self.Vinv = utils.fast_rank_one(self.Vinv, x_n)
            S += x_n * y_n
            theta = np.linalg.inv(self.V) @ S
            rec = np.argmax(self.X @ theta)
            self.arms_recommended.append(rec)
            self.pulled[best_idx] += 1
            if t%logging_period == 0:
                print('ts run', self.name, 'iter', t, "/", self.T, end="\r")
                d = {'recommended':int(rec==self._best_idx)}
                for i in range(self.X.shape[0]):
                    d[f'arm_{i}'] = self.pulled[i]
                wandb.log(d, step=t)



class TopTwoThetaAlgorithm(Linear):
    def __init__(self, X, Y, theta_star, T, sigma, name):
        super().__init__(X, Y, theta_star, T, sigma, name)
        self.Z = Y
        self.B = np.matmul(X.reshape(-1,self.d,1), X.reshape(-1,1,self.d))
        self.Vinv = np.linalg.inv(self.V)
        self.toptwo = []
        self.pulled = np.zeros(X.shape[0])
        self.k = 10
        self._best_idx = np.argmax(self.Z@theta_star)
        self.Vs = np.eye(len(self.theta_star))
        self.Vsinv = 2*np.eye(len(self.theta_star))
        
    def run(self, eta=10, logging_period=100):
        self.theta = np.ones(len(self.theta_star))#np.random.randn(len(self.theta_star))*10
        S = 0
        self.l = np.ones(self.X.shape[0])/self.X.shape[0]
        ada = AdaHedge(self.X.shape[0])
        for t in range(self.T):
            theta_1 = self.theta
            best_idx = np.argmax(self.Z@theta_1)
            theta_2s = []
            k = self.k
            while len(theta_2s) < 20:
                # draw k theta's and compute the best x at the same time to make it faster
                theta_2_mat = np.random.multivariate_normal(mean=self.theta, 
                                                            cov=self.Vinv, 
                                                            size=k)
                for tt in theta_2_mat:
                    if np.argmax(self.Z @ tt) != best_idx:
                        theta_2 = tt
                        theta_2s.append(tt) 
                k = 2*k        
                if k > 50000:
                    break
            if k > 50000:
                break

            g = 0
            for theta_2 in theta_2s:
                g += np.array([((theta_1 - theta_2).T@x)**2 for x in self.X])
            g = g/len(theta_2s)
            idx = np.random.choice(self.X.shape[0], p=self.l)
            x_n = self.X[idx]
            y_n = self.theta_star @ x_n + self.sigma * np.random.randn()


            self.V += np.outer(x_n, x_n)
            self.Vinv = utils.fast_rank_one(self.Vinv, x_n)
            
            # self.Vs += np.outer(x_n, x_n)*100/np.sqrt(self.T)
            # self.Vsinv = np.linalg.inv(self.Vs)

            S += x_n * y_n
            self.theta = self.Vinv @ S
            rec = np.argmax(self.Z @ self.theta)
            self.arms_recommended.append(rec)
            
            #r = g/np.sqrt(t+1)
            #self.l = self.l*np.exp(r - max(r))
            #self.l = self.l/np.sum(self.l)
            ada.incur(-g)
            self.l = ada.act()
            self.pulled += self.l

            if t%logging_period == 0:
                #print('t', t, 'norm', np.linalg.norm(g))
                #print(theta_2, theta_1, best_idx, best_idx_2)
                #print('l', self.l)
                print('toptwo run', self.name, 'iter', t, "/", self.T, end="\r")
                d = {'pulled':idx,'recommended':int(rec==self._best_idx)}
                for i in range(self.X.shape[0]):
                    d[f'arm_{i}'] = self.pulled[i]/(t+1)
                wandb.log(d, step=t)
        
        quit = len(self.arms_recommended)
        rec = self.arms_recommended[-1]
        if quit < self.T:
            for i in range(quit, self.T):
                self.arms_recommended.append(rec)



class TopTwoProjectionAlgorithm(Linear):
    def __init__(self, X, Y, theta_star, T, sigma, name):
        super().__init__(X, Y, theta_star, T, sigma, name)
        self.B = np.matmul(X.reshape(-1,self.d,1), X.reshape(-1,1,self.d))
        self.Vinv = np.linalg.inv(self.V)
        self.toptwo = []
        self.pulled = np.zeros(X.shape[0])
        self.k = 10
        self._best_idx = np.argmax(X@theta_star)
        
    def run(self, eta=10, logging_period=100):
        self.theta = np.ones(self.d)
        S = 0
        self.l = np.ones(self.X.shape[0])/self.X.shape[0]
        ada = AdaHedge(self.X.shape[0])
        for t in range(self.T):
            theta_1 = self.theta
            best_idx = np.argmax(self.X@theta_1)
            best_idx_2 = best_idx
            a = 0 
            while best_idx == best_idx_2 and a < 50000:
                # draw k theta's and compute the best x at the same time to make it faster
                theta_2_mat = np.random.multivariate_normal(mean=self.theta, 
                                                            cov=self.Vinv, 
                                                            size=self.k)
                for tt in theta_2_mat:
                    if np.argmax(self.X @ tt) != best_idx:
                        best_idx_2 = np.argmax(self.X @ tt)
                        theta_2 = tt
                        break
                a+=self.k
            if a >=50000:
                print('BROKE EARLY', self.name)
                break
            
            theta_2 = self.alt(best_idx_2, best_idx)
            g = np.array([((theta_1 - theta_2).T@x)**2 for x in self.X])
            idx = np.random.choice(self.X.shape[0], p=self.l)
            x_n = self.X[idx]#self.X[max_idx]
            y_n = self.theta_star @ x_n + self.sigma * np.random.randn()

            self.V += np.outer(x_n, x_n)
            self.Vinv = utils.fast_rank_one(self.Vinv, x_n)
            S += x_n * y_n
            self.theta = self.Vinv @ S
            rec = np.argmax(self.X @ self.theta)
            self.arms_recommended.append(rec)
            #self.l = (1-2/(t+3))*self.l + 2/(t+3)*np.eye(len(self.l))[np.argmax(g)] 
            #self.l = self.l*np.exp(20/np.sqrt(t+1)*g)
            #self.l = self.l/np.sum(self.l)
            ada.incur(-g)
            self.l = ada.act()
            self.pulled = self.l


            
            if t%logging_period == 0:
                A = np.array([self.X[0] - self.X[1],
                              self.X[0] - self.X[2]] )
                print('toptwo run', self.name, 'iter', t, "/", self.T, end="\r")
                p = utils.prob_region(A, self.theta, self.Vinv)
                rho = (t+1)/(-np.log(max(1-p,.01)))
                print('toptwo run', self.name, 'iter', t, "/", self.T, end="\r")
                d = {'pulled':idx,'recommended':int(rec==self._best_idx), 'rho':rho}
                for i in range(self.X.shape[0]):
                    d[f'arm_{i}'] = self.pulled[i]
                wandb.log(d, step=t)
        
        quit = len(self.arms_recommended)
        rec = self.arms_recommended[-1]
        if quit < self.T:
            for i in range(quit, self.T):
                self.arms_recommended.append(rec)

    def alt(self, best_idx_2, best_idx):
        xstar = self.X[best_idx]
        z = self.X[best_idx_2]
        A = sum([np.outer(self.X[i], self.X[i])*self.l[i] for i in range(self.X.shape[0])])
        A = np.linalg.inv(A)
        gap = (xstar-z)@self.theta
        var = (xstar-z).T@A@(xstar-z)
        a = self.theta - gap*A@(xstar-z)/var
        return a
    


class TopTwoAlgorithm(Linear):
        
    def __init__(self, X, Y, theta_star, T, sigma, name):
        super().__init__(X, Y, theta_star, T, sigma, name)
        self.B = np.matmul(X.reshape(-1,self.d,1), X.reshape(-1,1,self.d))
        self.Vinv = np.linalg.inv(self.V)
        self.toptwo = []
        self.pulled = []
        self.k = 10
        
    def run(self, logging_period=1):
        theta = np.zeros(self.d)
        S = 0
        
        for t in range(self.T):
            theta_1 = np.random.multivariate_normal(theta, self.Vinv)
            best_idx = np.argmax(self.X @ theta_1)
            x_1 = self.X[best_idx]

            # try it for a first time
            theta_2 = np.random.multivariate_normal(theta, self.Vinv)
            best_idx_2 = np.argmax(self.X@theta_2)

            a = 0 
            while best_idx == best_idx_2:
                # draw k theta's and compute the best x at the same time to make it faster
                theta_2_mat = np.random.multivariate_normal(mean=theta, 
                                                      cov=self.Vinv, size=self.k)
                max_x2_vec = np.argmax(self.X @ theta_2_mat.transpose(), axis=0)
                #print(max_x2_vec!=best_idx)
                if any(max_x2_vec!=best_idx): # if there is some index that is different
                    # find the first place where they are different
                    #print(np.where(max_x2_vec != best_idx)[0])
                    best_idx_2 = max_x2_vec[np.where(max_x2_vec != best_idx)[0][0]]
                a+=1
                if a > 10000:
                    break
            if a > 10000:
                    break
            
            x_2 = self.X[best_idx_2]
            self.toptwo.append([best_idx, best_idx_2])
            

            min_idx = np.argmin((x_1 - x_2) @ np.linalg.inv(self.V + self.B) @ (x_1 - x_2))
            self.pulled.append(min_idx)
            x_n = self.X[min_idx]
            y_n = self.theta_star @ x_n + self.sigma * np.random.randn()

            self.V += np.outer(x_n, x_n)
            self.Vinv = utils.fast_rank_one(self.Vinv, x_n)
            S += x_n * y_n
            theta = self.Vinv @ S
            self.theta = theta
            self.arms_recommended.append(np.argmax(self.X @ theta))

            if t%logging_period == 0:
                print('toptwo run', self.name, 'iter', t, "/", self.T, end="\r")
        
        quit = len(self.arms_recommended)
        rec = self.arms_recommended[-1]
        if quit < self.T:
            for i in range(quit, self.T):
                self.arms_recommended.append(rec)

                    
class XYStatic(Linear):
    
    def __init__(self, X, Y, theta_star, T, sigma, name):
        super().__init__(X, Y, theta_star, T, sigma, name)
        #self.Y = compute_Y(X)
        self.Vinv = np.linalg.inv(self.V)
        self.pulled = np.zeros(X.shape[0])
        self._best_idx = np.argmax(X@theta_star)

    def run(self, logging_period=1):
        lam_f,_,_ = utils.FW(self.X, self.Y, iters=1000)
        del self.Y
        S = 0
        
        for t in range(self.T):
            idx = np.random.choice(self.n, p=lam_f)
            self.pulled[idx] += 1
            x_n = self.X[idx]
            y_n = x_n @ self.theta_star + self.sigma*np.random.randn()
            self.V += np.outer(x_n, x_n)
            self.Vinv = utils.fast_rank_one(self.Vinv, x_n)
            S += x_n * y_n
            theta = np.linalg.inv(self.V) @ S 
            rec = np.argmax(self.X @ theta)
            self.arms_recommended.append(rec)
            d = {'pulled':idx,'recommended':int(rec==self._best_idx)}
            for i in range(self.X.shape[0]):
                d[f'arm_{i}'] = self.pulled[i]/(t+1)
            wandb.log(d, step=t)        
            if t%logging_period == 0:
                print('xy static run', self.name, 'iter', t, "/", self.T, end="\r")

    
class XYAdaptive(Linear):
    
    def __init__(self, X, Y, theta_star, T, sigma, name):
        super().__init__(X, Y, theta_star, T, sigma, name)
        self.k = 5
        self.Vinv = np.linalg.inv(self.V)
        #self.Y = compute_Y(X)
        # self.k = k #TODO: add this later

    def run(self, logging_period=1, FW_verbose=False, FW_logging_period=100):
        S = 0
        lam_f = np.ones(self.n)/self.n
        theta = np.zeros(self.d)
        for t in range(self.T):
            theta_mat = np.random.multivariate_normal(mean=theta, 
                                                      cov=self.Vinv, size=self.k)
            max_x_vec = np.argmax(self.X @ theta_mat.transpose(), axis=0)  
            # this should be dimension k
            
            X_t = self.X[max_x_vec]
            Y_t = compute_Y(X_t)
            lam_f, _, _ = FW(self.X, Y_t, initial=lam_f, 
                             iters=20, step_size=1, 
                             logging_step=FW_logging_period, verbose=FW_verbose)
            
            ind_n = np.random.choice(self.X.shape[0], p=lam_f)
            
            x_n = self.X[ind_n]
            y_n = x_n @ self.theta_star + self.sigma * np.random.randn()
            
            self.V += np.outer(x_n, x_n)
            self.Vinv = np.linalg.inv(self.V)
            S += x_n * y_n
            theta = self.Vinv @ S
            self.arms_recommended.append(np.argmax(self.X @ theta))
            
            if t%logging_period == 0:
                print('xy adaptive run', self.name, 'iter', t, "/", self.T, end="\r")



class GeneralTopTwoLinear(Linear):
    def __init__(self, X, Y, gen_star, T, sigma, name):
        super().__init__(X, Y, gen_star, T, sigma, name)
        self.pi = Gaussian(np.zeros(self.d), self.V)
        if type(gen_star) is np.ndarray:
            self.gen_star = GenericFunction(lambda x: x@gen_star, sigma)
        else:
            self.gen_star = gen_star
        self.B = 20
        self.pulled = []
        self.name = name
        self.delta=.0001
        
    def run(self, logging_period=1, k=10):
        for t in range(self.T):

            f1 = self.pi.sample()
            idx1 = np.argmax(f1.evaluate(self.X))
            x1 = self.X[idx1]

            a = 0
            idx2 = idx1
            while idx1 == idx2:
                f2s = self.pi.sample(k)  # TODO: sample 10 at a time
                for f2 in f2s:
                    idx2 = np.argmax(f2.evaluate(self.X))
                    if idx1 != idx2:
                        break
                a+=k
                if a > 1/self.delta:
                    break
            if a > 1/self.delta:
                break
                    
            x2 = self.X[idx2]

            v = []               
            for idx in range(self.n):
                x = self.X[idx]
                expected_diff = 0
                expected_diff_squared = 0
                
                for b1 in range(self.B):
                    gen_b1 = self.pi.sample()
                    y_b1 = gen_b1.pull(x)
                    weight = t # we should change this weighting to simulating t observations from x
                    pi_plus = self.pi.update_posterior(x*weight, y_b1, copy=True)
                    gen_b2 = pi_plus.sample()
                    expected_diff += ( gen_b2.evaluate(x1) - gen_b2.evaluate(x2) ) 
                    expected_diff_squared += ( gen_b2.evaluate(x1) - gen_b2.evaluate(x2) )**2
                v.append( expected_diff_squared/self.B - (expected_diff/self.B)**2 )
                
            min_idx = np.argmin(v)
            self.pulled.append(min_idx)
            x_n = self.X[min_idx]
            y_n = self.gen_star.pull(x_n)
            self.pi.update_posterior(x_n, y_n)
            
            fhat = self.pi.map()
            idx = np.argmax(fhat.evaluate(self.X))
            self.arms_recommended.append(idx)

            if t%logging_period == 0:
                print('general run', self.name, 'iter', t, "/", self.T, end="\r")
        quit = len(self.arms_recommended)
        rec = self.arms_recommended[-1]
        if quit < self.T:
            for i in range(quit, self.T):
                self.arms_recommended.append(rec)




class AdaHedge:
    def __init__(self, K):
        self.L = np.zeros(K)
        self.delta = 0.01

    def act(self):
        eta = np.log(len(self.L)) / self.delta
        u = np.exp(-eta * (self.L - np.min(self.L)))
        return u / np.sum(u)

    def incur(self, l):
        u = self.act()
        eta = np.log(len(self.L)) / self.delta
        self.L += l
        m = np.min(l) - 1 / eta * np.log(u.T @ np.exp(-eta * (l - np.min(l))))
        # println(η, " ", u, " ", u.T @ l, " ", -1 / η * np.log(u.T @ np.exp(-η * l)));
        #@assert isfinite(m) and u.T @ l >= m - 1e-7 "hedge loss $(u.T @ l), Δ=$(self.delta), η=$eta, mix loss $m, u=$u, L=$(self.L)";
        self.delta += u.T @ l - m
