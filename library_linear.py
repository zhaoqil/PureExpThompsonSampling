import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from bandit_type import Linear
from utils import *
import utils

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
                # wandb.log(d, step=t)



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
                if k > 150000:
                    break
            if k > 150000:
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

            S += x_n * y_n
            self.theta = self.Vinv @ S
            rec = np.argmax(self.Z @ self.theta)
            self.arms_recommended.append(rec)
            
            ada.incur(-g)
            self.l = ada.act()
            self.pulled += self.l
            
            if t%logging_period == 0:
                print('toptwo run', self.name, 'iter', t, "/", self.T, end="\r")
                d = {'pulled':idx,'recommended':int(rec==self._best_idx)}
                for i in range(self.X.shape[0]):
                    d[f'arm_{i}'] = self.pulled[i]/(t+1)
        
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
        self.delta += u.T @ l - m
