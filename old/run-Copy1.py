import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import multiprocessing as mp
import contextualbandits.library_concept as library_concept
from importlib import reload
import argparse

reload(library_concept)

def worker(algorithm, X, theta_star, T, sigma, name):
    np.random.seed()
    algorithm_instance = algorithm(X, theta_star, T, sigma, name)
    algorithm_instance.run()
    return algorithm_instance.arm_sequence


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int)
    parser.add_argument('--reps', type=int)
    args = parser.parse_args()
    
    
    T = args.T  # time steps
    reps = args.reps  # repetitions of the algorithm
    
    d = 10  # dimension of arms
    K = 100  # number of arms
    X = np.random.randn(K, d)  # arms
    
    theta_star = np.random.randn(d)  # 
    idx_star = np.argmax(X@theta_star)  # index of best arm
    
    pool = mp.Pool(40)
    
    algorithms = [library_concept.TopTwoAlgorithm, library_concept.ThompsonSampling, library_concept.XYStatic, library_concept.XYAdaptive]
    
    for algorithm in algorithms:
        args = [(algorithm, X, theta_star, T, 1, i) for i in range(reps)]
        results = pool.starmap(worker, args)
        m1 = (results == idx_star).mean(axis=0)
    
    args1 = [(library_concept.TopTwoAlgorithm, X, theta_star, T, 1, i) for i in range(reps)]
    args2 = [(library_concept.ThompsonSampling, X, theta_star, T, 1, i) for i in range(reps)]
    args3 = [(library_concept.XYStatic, X, theta_star, T, 1, i) for i in range(reps)]
    args4 = [(library_concept.XYAdaptive, X, theta_star, T, 1, i) for i in range(reps)]
    
    results1 = pool.starmap(worker, args1)
    results2 = pool.starmap(worker, args2)
    results3 = pool.starmap(worker, args3)
    results4 = pool.starmap(worker, args4)
    
    m1 = (results1 == idx_star).mean(axis=0)
    m2 = (results2 == idx_star).mean(axis=0)
    m3 = (results3 == idx_star).mean(axis=0)
    m4 = (results4 == idx_star).mean(axis=0)
    
    s1 = (results1 == idx_star).std(axis=0)/np.sqrt(reps)
    s2 = (results2 == idx_star).std(axis=0)/np.sqrt(reps)
    s3 = (results3 == idx_star).std(axis=0)/np.sqrt(reps)
    s4 = (results4 == idx_star).std(axis=0)/np.sqrt(reps)
    
    xaxis = np.arange(T)

    plt.plot(xaxis, m1)
    plt.fill_between(xaxis, m1 - 1.96 * s1, m1 + 1.96 * s1,
                     color='blue', alpha=0.2, label='Top two posterior')
    
    plt.plot(xaxis, m2)
    plt.fill_between(xaxis, m2 - 1.96 * s2, m2 + 1.96 * s2,
                     color='orange', alpha=0.2, label='Thompson sampling')
    
    plt.plot(xaxis, m3)
    plt.fill_between(xaxis, m3 - 1.96 * s3, m3 + 1.96 * s3,
                     color='green', alpha=0.2, label='XY static')
    
    plt.plot(xaxis, m4)
    plt.fill_between(xaxis, m4 - 1.96 * s4, m4 + 1.96 * s4,
                     color='red', alpha=0.2, label='XY adaptive')
    
    plt.xlabel('time')
    plt.ylabel('identification rate')
    
    plt.legend()
    plt.savefig('results.png')