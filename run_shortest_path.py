# TODO: WORK IN PROGRESS.

num_trials = 1
gaps = 0.2

for trial in range(num_trials):
            
    G,source, sink, thetastar = generate_divided_net_sparse(26, 2, diff = gap, num_paths = 2)
    mo = ShortestPathDAGOracle(G,source,sink)

    params.append((thetastar, mo, np.random.randint(1e6), experiment_num, False, False))
            
    experiment_num += 1
    print(f'thetastar {thetastar}')

def paths_experiment_divided_net_vary_gap(num_cores, num_trials = 5):
    print("running paths experiment divided net style vary gap....")

    gaps = [.2,.15,.1,.05]

    params = []
    experiment_num = 0
    for gap in gaps: 
        print(f"gap {gap}")
        for trial in range(num_trials):
            
            G,source, sink, thetastar = generate_divided_net_sparse(26, 2, diff = gap, num_paths = 2)
            mo = ShortestPathDAGOracle(G,source,sink)

            params.append((thetastar, mo, np.random.randint(1e6), experiment_num, False, False))
            
            experiment_num += 1
            print(f'thetastar {thetastar}')

            
    pool = mp.Pool(num_cores)
    output = pool.starmap(run_experiment, params)

    filename = './results/{}_{}.pkl'.format(time.time(), 'shortest_path_divided_net_vary_gap')
    with open(filename, 'wb') as f:
        pickle.dump(output, f) 