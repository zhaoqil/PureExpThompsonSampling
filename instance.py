import numpy as np
import distribution
from itertools import combinations
import h5py
import pandas as pd
from itertools import product, combinations

import matplotlib.pyplot as plt
import logging
import sys
import networkx as nx
import random
import itertools

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
    print(X.shape, theta_star.shape)
#     # Open the HDF5 file
#     with h5py.File("linear_instance.h5", "r") as f:
#         theta = f["theta"]
#         arms = f["arms"]
#     print(theta, arms)

    # Print the arrays
#     print(a)
#     print(b)
#     X = 
#     X = np.random.randn(K, d)
#     norms = np.linalg.norm(X, axis=1).reshape(K, 1)
#     X /= norms
#     min_pair = []
#     min_norm = 10
#     for i in range(len(X)):
#         for j in range(i+1, len(X)):
#             if np.linalg.norm(X[i] - X[j]) < min_norm:
#                 min_pair = [i,j]
#                 min_norm = np.linalg.norm(X[i] - X[j])
            
            
#     theta_star = X[min_pair[0]] + .01*(X[min_pair[1]] - X[min_pair[0]])
#     #np.random.randn(d)
#     #theta_star = theta_star/np.linalg.norm(theta_star)
#     return X, theta_star
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
    
    # theta = np.random.rand(d)    
    #theta_star = pd.read_csv('theta_topm.csv', header=None).to_numpy().squeeze()
    theta_star = np.array([1 - i*alpha for i in range(d)])
    
    return np.eye(d), theta_star, indicator_vectors


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
    elif name=='mvt':
        f = generate_mvt
    return f(**params)

def mvt(d, k):
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
            a[i,j] = 1*alpha2
            x.extend(a.reshape(-1))
        X.append(x)
    X = np.array(X)
    thetastar = np.zeros(X.shape[1])
    idx1 = 1
    idx2 = 1
    idx3 = 1
    thetastar[d*k+idx1*idx2] = .8
    thetastar[d*k+2*k**2+idx2*idx3] = .05
    return X, thetastar

def generate_mvt(d, k):
    design = []
    for seq in product(range(k), repeat=d):
        row = [1] + list(seq)
        for i,j in combinations(range(d),2):
            row.append(seq[i]*seq[j])
        design.append(row)
    print(design, len(design))
    design = np.array(design)
    thetastar = np.zeros(design.shape[1])
#     idx1 = 1
#     idx2 = 1
#     idx3 = 1
#     thetastar[0] = 1
#     thetastar[1] = 1
#     thetastar[3] = 0.95
#     thetastar[4] = 0.01
#     thetastar[6] = 0.005
    thetastar = np.array([0, 0.015, 0.995, 1, 0, 0, 0])
    
#     thetastar[d-1] = 1
#     thetastar[d] = 0.9

#     thetastar[d+1] = 0.8
#     thetastar[2*d] = 0.1
#     thetastar[k*d] = 0.5
#     thetastar[d*k+idx1*idx2] = .8
#     thetastar[d*k+2*k**2+idx2*idx3] = .05
    return design, thetastar

def generate_mvt_2(d, k):
    thetastar = np.array([0, 0.01, 0.995, 1, 0, 0, 0])
    X = [[1, 1, 0, 0, 0, 0, 0], [1, 0, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0]]
    X = np.array(X)
    return X, thetastar

# def createDividedFeedforwardGraph(layer_num, layer_width):
#     '''
#     Creates feedforward network
    
#     input:
#         layer_num: number of layers
#         layer_width: width of the layers
#         edge_prob: probability that edge b/w intermediate layers is kept
        
#     output:
#         G: graph 
#         source and sink
#     '''

#     G = nx.DiGraph()

#     #create nodes and group into layers
#     layers = {}
#     layers[0] = {}
#     layers[1] = {}
    
#     source = 0
#     sink = 1
#     G.add_node(source)
#     G.add_node(sink)
#     node_counter = 2
    
#     for graph_part in [0,1]:

#         for i in range(layer_num):
#             layers[graph_part][i] = []
#             for j in range(layer_width):

#                 layers[graph_part][i].append(node_counter)
#                 G.add_node(node_counter)
#                 node_counter += 1

#         #add_edges
#         for i in layers[graph_part][0]:
#             G.add_edge(source, i)

#         for i in layers[graph_part][layer_num-1]:
#             G.add_edge(i, sink)

#         for cur_layer in range(layer_num-1):
#             for node_prev in layers[graph_part][cur_layer]:
#                 for node_next in layers[graph_part][cur_layer+1]:
#                     G.add_edge(node_prev,node_next)

#     return G, source, sink, layers

# def generate_divided_net_sparse(layer_num, layer_width, diff = .1, num_paths = 5):
    
#     G, source, sink, layers = createDividedFeedforwardGraph(layer_num, layer_width)
#     mo = ShortestPathDAGOracle(G,source,sink)
#     d = len(mo.edgelist)
#     thetastar = np.zeros((d,))
    
#     edge1 =  mo.edge_to_idx[(0,layers[0][0][0])]
#     weights = np.ones(d)
#     weights[edge1] = 1000000000
#     val, new_good_z = mo.max(weights)

#     np.putmask(thetastar,new_good_z.astype(int),1)
    
#     edge2 = mo.edge_to_idx[(0,layers[1][0][0])]
#     weights = np.ones(d)
#     weights[edge2] = 1000000000
#     val, new_good_z = mo.max(weights)

#     np.putmask(thetastar,new_good_z.astype(int),1-diff)
    
#     np.putmask(thetastar,thetastar == 0, -1)
        
#     return G, source, sink, thetastar