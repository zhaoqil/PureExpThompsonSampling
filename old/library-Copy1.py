import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import multiprocessing as mp


class TopTwoAlgorithm(object):
    def __init__(self, X, theta_star, T, sigma=1, name=""):
        self.X = X
        self.d = X.shape[1]
        self.theta = np.zeros(self.d)
        self.theta_star = theta_star
        self.B = np.matmul(X.reshape(-1,self.d,1), X.reshape(-1,1,self.d))
        self.T = T
        self.V = np.eye(self.d)
        self.sigma = sigma
        self.name = name
        self.arm_sequence = []  # TODO: change this variable name
        
    def run(self, logging_period=1, verbose=False):
        errs = []
        S = 0
        for t in range(self.T):
            theta_1 = np.random.multivariate_normal(self.theta, self.V)
            best_idx = np.argmax(self.X@theta_1)
            x_1 = self.X[best_idx]
            
            best_idx_2 = best_idx
            while best_idx == best_idx_2:
                theta_2 = np.random.multivariate_normal(self.theta, self.V)
                best_idx_2 = np.argmax(self.X@theta_2)
                x_2 = self.X[best_idx_2]

            min_idx = np.argmin((x_1 - x_2) @ np.linalg.inv(self.V + self.B) @ (x_1 - x_2))
            x_n = self.X[min_idx]
            y_n = self.theta_star @ x_n + self.sigma * np.random.randn()

            self.V += np.outer(x_n, x_n)
            S += x_n * y_n
            self.theta = np.linalg.inv(self.V) @ S
            
            errs.append(np.linalg.norm(self.theta - self.theta_star))
            
            if t%logging_period == 0:
                # print('run', self.name, 'iter', t,'\n')
                self.arm_sequence.append(np.argmax(self.X @ self.theta))
                if verbose: 
                    plt.xlabel('iteration')
                    plt.ylabel(r'$\|\theta_*-\hat{\theta}\|$', rotation=0, labelpad=30)
                    plt.plot(errs);
                    plt.show()
                    clear_output(wait=True)
    
    def pi(self, theta, V, idx, repeat=10000):
        '''
        Probability that this idx is the best
        '''
        x_star = self.X[idx]
        count = 0        
        for _ in range(repeat):
            random_theta = np.random.multivariate_normal(theta, V)
            count += (idx == np.argmax(X@theta))
        return count / repeat

#     @staticmethod
#     def gap(self, x, x_star=self.x_star):
#         return (x_star - x) @ theta_star


def A(X, lambda_):
    return X.T@np.diag(lambda_)@X


class ThompsonSampling(object):

    def __init__(self, X, theta_star, T, sigma=1, name=""):
        self.X = X
        self.d = X.shape[1]
        self.sigma = sigma
        self.arm_sequence = []
        self.V = np.eye(self.d)
        self.theta_star = theta_star
        self.name = name
        self.T = T

    def run(self, logging_period=1, verbose=False):
        theta = np.zeros(self.d)
        S = 0
        errs = []
        for t in range(self.T):
            theta_hat = np.random.multivariate_normal(theta, np.linalg.inv(self.V))
            best_idx = np.argmax(self.X @ theta_hat)
            x_n = self.X[best_idx]
            y_n = x_n @ self.theta_star + self.sigma*np.random.randn()
            
            self.V += np.outer(x_n, x_n)
            S += x_n * y_n
            theta = np.linalg.inv(self.V) @ S         
            
            errs.append(np.linalg.norm(theta - self.theta_star))
            
            if t%logging_period == 0:
                # print('run', self.name, 'iter', t,'\n')
                self.arm_sequence.append(np.argmax(self.X@theta))
                if verbose: 
                    plt.xlabel('iteration')
                    plt.ylabel(r'$\|\theta_*-\hat{\theta}\|$', rotation=0, labelpad=30)
                    plt.plot(errs);
                    plt.show()
                    clear_output(wait=True)
                    
                    
class XYStatic(object):
    
    def __init__(self, X, theta_star, T, sigma=1, name=""):
        self.X = X
        self.n, self.d = X.shape
        self.sigma = sigma
        self.arm_sequence = []
        self.V = np.eye(self.d)
        self.theta_star = theta_star
        self.name = name
        self.T = T
        self.Y = self.compute_Y(X)
        
        
    def run(self, logging_period=1):
        lam_f = XYStatic.frank_wolfe(self.X, self.Y, self.grad_f)
        self.arm_sequence = np.random.choice(self.n, self.T, p=lam_f)
    
    @staticmethod
    def calc_max_mat_norm(Y, A_inv):
        n = Y.shape[0]
        res = np.zeros(n)
        for i in range(n):
            y = Y[i]
            res[i] = y.T @ A_inv @ y
        ind = np.argmax(res)
        return res[ind], ind
        
        
    @staticmethod
    def f(X, Y, lam):
        d = X.shape[1]
        A = X.T @ np.diag(lam) @ X + 0.0001 * np.eye(d)
        A_inv = np.linalg.inv(A)
        res, ind = XYStatic.calc_max_mat_norm(Y, A_inv)
        return res, ind, A_inv

    @staticmethod
    def grad_f(X, Y, lam):
        # TODO: better variable name for X
        _, ind, A_inv = XYStatic.f(X, Y, lam)
        return -np.power(Y @ A_inv @ Y[ind], 2)
        
        
    @staticmethod
    def frank_wolfe(X, Y, grad_f, N=500):
        n = X.shape[0]
        d = X.shape[1]
        inds = np.random.choice(n, 2*d, p=1/n * np.ones(n)).tolist()
        lam = np.bincount(inds, minlength=n) / len(inds)
        if sum(lam) != 1:
            lam[np.argmax(lam)] += (1 - sum(lam))
        for i in range(2*d+1, N+1):
            eta = 2/(i+1)
            ind = np.argmin(grad_f(X, Y, lam))
            lam = (1-eta)*lam + eta * np.eye(1, n, ind).flatten()
            if sum(lam) != 1:
                lam[np.argmax(lam)] += (1 - sum(lam))
            inds.append(ind)
        return lam
    
    @staticmethod
    def compute_Y(X):
        #TODO: change it to one-line
        res = []
        for i in range(X.shape[0]):
            for j in range(i+1, X.shape[1]):
                res.append(X[i] - X[j])
        return np.array(res)

class XYAdaptive(object):
    
    def __init__(self, X, theta_star, T, sigma=1, name=""):
        self.X = X
        self.d = X.shape[1]
        self.theta = np.zeros(self.d)
        self.sigma = sigma
        self.arm_sequence = []
        self.V = np.eye(self.d)
        self.theta_star = theta_star
        self.name = name
        self.T = T
        self.k = 20
        self.Y = XYStatic.compute_Y(X)
        # self.k = k #TODO: add this later
        
        
        
    def run(self, logging_period=1):
        S = 0
        for t in range(self.T):
            theta_mat = np.random.multivariate_normal(mean=self.theta, cov=self.V, size=self.k)
            max_x_vec = np.argmax(self.X @ theta_mat.transpose(), axis=0)  # this should be dimension k
            
            X_t = self.X[max_x_vec]
            Y_t = XYStatic.compute_Y(X_t)
            lam_f = XYStatic.frank_wolfe(X_t, Y_t, XYStatic.grad_f)
            ind_n = np.random.choice(X_t.shape[0], 1, p=lam_f)
            
            x_n = X_t[ind_n][0]
            y_n = x_n @ self.theta_star + self.sigma * np.random.randn()
            
            self.V += np.outer(x_n, x_n)
            S += x_n * y_n
            self.theta = np.linalg.inv(self.V) @ S
            
            if t%logging_period == 0:
                self.arm_sequence.append(np.argmax(self.X @ self.theta))

            