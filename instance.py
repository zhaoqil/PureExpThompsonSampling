import numpy as np
import distribution
from itertools import combinations

def sphere(K, d):
    '''
    Generates arms and optimal arm of a unit sphere instance.
    K: int, number of arms
    d: int, dimension of each arm
    Returns:
    - X: numpy array, matrix of K arms of dimension d.
    - theta_star: numpy array, 1 x d vector representing the optimal arm.
    '''
    X = np.random.randn(K, d)
    norms = np.linalg.norm(X, axis=1).reshape(K, 1)
    X /= norms
    min_pair = []
    min_norm = 10
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if np.linalg.norm(X[i] - X[j]) < min_norm:
                min_pair = [i,j]
                min_norm = np.linalg.norm(X[i] - X[j])
            
            
    theta_star = X[min_pair[0]] + .01*(X[min_pair[1]] - X[min_pair[0]])
    #np.random.randn(d)
    #theta_star = theta_star/np.linalg.norm(theta_star)
    return X, theta_star

def soare(d, alpha):
    '''
    Generates arms and optimal arm of a "Soare" instance.
    d: int, dimension of each arm
    alpha: float, 
    Returns:
    - X: numpy array, matrix of d+1 arms of dimension d.
    - 2*e_1: numpy array, 1 x d vector representing the optimal arm.
    '''
    X = np.eye(d)
    e_1, e_2 = X[:2]
    x_prime = np.cos(alpha) * e_1 + np.sin(alpha) * e_2
    X = np.concatenate([X, np.array([x_prime])])
    return X, e_1


def topk(d,k):
    np.random.seed(10)
    # generate all combinations of indices
    all_combinations = list(combinations(range(d), k))
    
    # create an array of zeros
    indicator_vectors = np.zeros((len(all_combinations), d))
    
    # set the relevant indices to one
    for i, indices in enumerate(all_combinations):
        indicator_vectors[i, indices] = 1
    
    theta = np.random.rand(d)    
    
    return np.eye(d), theta, indicator_vectors


def MAB(d):
    '''
    Generates arms and optimal arm of a MAB instance.
    d: int, dimension of each arm
    alpha: float, 
    Returns:
    - X: numpy array, matrix of d+1 arms of dimension d.
    - 2*e_1: numpy array, 1 x d vector representing the optimal arm.
    '''
    X = np.eye(d)
    theta = np.array([.5, .4, .3, .2, .1])#.5 for i in range(1,d+1)])
    return X, theta

def entropy(K):
    f_star = distribution.GenericFunction(lambda x: -x*np.log(x) - (1-x)*np.log(1-x), .1)
    return np.random.rand(K).reshape(-1,1), f_star

def get_instance(name, params):
    if name=='soare':
        f = soare
    elif name=='sphere':
        f = sphere
    elif name=='entropy':
        f = entropy
    elif name=='MAB':
        f = MAB
    elif name=='topk':
        f = topk
    return f(**params)
