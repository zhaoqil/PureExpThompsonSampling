import numpy as np
from itertools import combinations
import pandas as pd

def sphere(K, d):
    '''
    Generates arms and optimal arm of a unit sphere instance.
    K: int, number of arms
    d: int, dimension of each arm
    Returns:
    - X: numpy array, matrix of K arms of dimension d.
    - theta_star: numpy array, 1 x d vector representing the optimal arm.
    '''
    np.random.seed(10)
    theta_star = pd.read_csv('theta.csv', header=None).to_numpy().squeeze()
    X = pd.read_csv('arms.csv', header=None, delimiter='\t').to_numpy()
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


def topk(d, k):
    alpha = .05
    np.random.seed(10)
    # generate all combinations of indices
    all_combinations = list(combinations(range(d), k))
    
    # create an array of zeros
    indicator_vectors = np.zeros((len(all_combinations), d))
    
    # set the relevant indices to one
    for i, indices in enumerate(all_combinations):
        indicator_vectors[i, indices] = 1
        
    theta_star = np.array([1 - i*alpha for i in range(d)])
    
    return np.eye(d), theta_star, indicator_vectors

def mvt(d,k):
    eye = np.eye(k)
    X = []
    alpha1 = 1
    alpha2 = 1
    for seq in product(range(k), repeat=d):
        x = [1]
        for i in seq:
            x.extend(eye[i])
        for idx1,idx2 in combinations(range(d), 2):
            i,j = seq[idx1], seq[idx2]
            a = np.zeros((k,k))
            #print(idx1, idx2, i,j, a.shape)
            a[i,j] = 1*alpha2
            x.extend(a.reshape(-1))
        X.append(x)
    X = np.array(X)
    np.random.seed(50000)
    thetastar = np.zeros(X.shape[1])
    idx1 = 1
    idx2 = 1
    idx3 = 1
    thetastar[d*k+idx1*idx2] = .8
    thetastar[d*k+2*k**2+idx2*idx3] = .05
    np.random.seed()
    return X, thetastar


def get_instance(name, params):
    if name=='soare':
        f = soare
    elif name=='sphere':
        f = sphere
    elif name=='topk':
        f = topk
    elif name=='mvt':
        f = mvt
    return f(**params)
