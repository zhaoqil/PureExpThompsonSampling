import numpy as np
import matplotlib.pyplot as plt
import distribution
import library_linear
from scipy.stats import multivariate_normal as mvn

def A(X, lambda_):
    return X.T @ np.diag(lambda_) @ X

def calc_max_mat_norm(Y, A_inv):
    n = Y.shape[0]
    res = np.zeros(n)
    for i in range(n):
        y = Y[i]
        res[i] = y.T @ A_inv @ y
    ind = np.argmax(res)
    return res[ind], ind

def compute_Y(X):
#     return np.concatenate([X[i]-X[i+1:] for i in range(X.shape[0]-1)], axis=0)
    n = X.shape[0]
    Y = []
    for i in range(n-1):
        for j in range(i+1, n):
            Y.append(X[i] - X[j])
    return np.array(Y)

def FW(X, Y, reg_l2=0, iters=500, 
       step_size=1, logging_step = 10000,
       verbose=False, initial=None):
    n, d = X.shape
    I = np.eye(n)
    if initial is not None:
        design = initial
    else:
        design = np.ones(n)
        design /= design.sum()  
    eta = step_size
    grad_norms = []
    history = []
    
    for count in range(1, iters):
        A_inv = np.linalg.pinv(X.T@np.diag(design)@X + reg_l2*np.eye(d))        
        #rho = np.array([y.T@A_inv@y for y in Y])
        rho = np.matmul(np.matmul(Y.reshape(-1, 1,d), A_inv), 
                        Y.reshape(-1, d,1)).reshape(Y.shape[0],)
        y_opt = Y[np.argmax(rho),:]
        g = y_opt @ A_inv @ X.T
        g = -g * g
        
        eta = step_size/(count+2)
        imin = np.argmin(g)
        design = (1-eta)*design+eta*I[imin]
        grad_norms.append(np.linalg.norm(g - np.sum(g)/n*np.ones(n)))
        
        if verbose and count % (logging_step) == 0:
            history.append(np.max(rho))
            fig, ax = plt.subplots(1,2)
            ax[0].plot(grad_norms)
            ax[1].plot(design)
            plt.show()
        
    return design, rho, history


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


def fast_rank_one(B,v):
    x = B@v
    return B - np.outer(x,x) / (1 + v.T @ B @ v)



def get_alg(alg, X, Y, f_star, T, sigma, ix):
        name = alg['name'] + f'_{ix}'
        cls = alg['alg_class']
        params = alg['params']
        if cls == 'ThompsonSampling':
            return library_linear.ThompsonSampling(X, Y, f_star, T, sigma, name)
        elif cls == 'TopTwoThetaAlgorithm':
            return library_linear.TopTwoThetaAlgorithm(X, Y, f_star, T, sigma, name)
        
def prob_region(A, theta, Sigma):
    mu = -A@theta
    Sigma = A@Sigma@A.T
    return mvn.cdf(np.zeros(len(mu)), mean=mu, cov=Sigma)