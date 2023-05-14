import sys
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

import library_concept
import library_linear
import instance

import utils
import json
import wandb

    

def worker(alg, X, Y, f_star, T, sigma, ix):
    print('algorithm', alg, 'name', alg['name'])
    np.random.seed()
    wandb.init(project="TTTS", 
               name="experiment_run_{}".format(ix), 
               group="experiment", 
               config={ 
                   "alg": alg, 
               })
    algorithm_instance = utils.get_alg(alg, X, Y, f_star, T, sigma, ix)
    algorithm_instance.run(logging_period=100)
    wandb.finish()
    return algorithm_instance.arms_recommended, algorithm_instance.pulled

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--T', type=int)
    #parser.add_argument('--reps', type=int)
    #parser.add_argument('--cpu', default=10, type=int)
    parser.add_argument('--parallelize', default='mp', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--path', default=os.getcwd(), type=str)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        params = json.load(f)
        print(params)


    T = params['global']['T']  # time steps
    sigma = params['global']['sigma']
    reps = params['global']['reps']  # repetitions of the algorithm
    cpu = params['global']['cpu']
    exp_name = params['global']['exp_name']
    path = args.path
    print('OUR PATH', path)



    inst = instance.get_instance(**params['global']['instance'])
    if len(inst) == 2:
        X, f_star = inst
        Z = X
    elif len(inst)==3:
        X, f_star, Z = inst

    algorithms = [alg for alg in params['algs'] if alg['active']]
    runs = []
    
    for alg in algorithms:
        runs += [(alg, X, Z, f_star, T, sigma, i) for i in range(reps)]

    if params['global']['parallelize'] == 'mp':
        pool = mp.Pool(cpu, maxtasksperchild=1000)
        all_results = pool.starmap(worker, runs)
    else:
        import ray
        ray.init(address='128.208.6.83:6379',)
        worker = ray.remote(scheduling_strategy="SPREAD")(worker)
        all_results = ray.get([worker.remote(*a) for a in runs])
        # all_results = []
        # for a in runs:
        #     all_results.append(worker(*a))

    
    idx_star = np.argmax(Z@f_star)#np.argmax(f_star.evaluate(X))  # index of best arm
    xaxis = np.arange(T)
    
    d = {}
    d_alg = {}
    d['idx_star'] = idx_star
    d['reps'] = reps
    d['xaxis'] = xaxis
    for i,alg in enumerate(algorithms):
        results = all_results[reps*i: reps*(i+1)]
        results_ar = np.array([r[0] for r in results])
        results_p = [np.array(r[1]) for r in results]
        d_alg[alg['name']] ={'results_ar':results_ar, 'results_p':results_p}
        plt.subplot(1,2,1)
        m = (results_ar == idx_star).mean(axis=0)
        s = (results_ar == idx_star).std(axis=0)/np.sqrt(reps)
        plt.plot(xaxis, m)
        plt.fill_between(xaxis, m - s, m + s, alpha=0.2, label=alg['alg_class'])
        plt.subplot(1,2,2)
        plt.plot(np.mean(results_p, axis=0))
        d['algs'] = d_alg
    with open(path+f'/results_{exp_name}.pkl', 'wb') as f:
        pickle.dump(d,f)

    plt.xlabel('time')
    plt.ylabel('identification rate')
    #plt.title(f'K {K} d {d}')
    plt.legend()
    plt.savefig(path+'/results.png')