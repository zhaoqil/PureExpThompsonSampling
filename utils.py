import numpy as np
import matplotlib.pyplot as plt
import library_linear
from scipy.stats import multivariate_normal as mvn

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