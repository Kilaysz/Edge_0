# SKENARIO 6 - PERBANDINGAN NON PU VS PU

import time
import logging
import pandas as pd
import numpy as np

from algo.qpso import QPSOServerPlacer
from utils.utils_all import DataUtils
from utils.utils_potential import DataUtilsPotential

def run_qpso_trial(base_stations, distances, k, max_iter, swarm_size, beta, mu, repeat=5):
    n = len(base_stations)
    fitness_list = []
    distance_list = []
    workload_var_list = []

    for _ in range(repeat):
        qpso = QPSOServerPlacer(
            base_stations=base_stations,
            distances=distances,
            beta=beta,
            max_iter=max_iter,
            swarm_size=swarm_size,
            mu=mu
        )
        qpso.place_server(n, k)
        fitness_list.append(qpso.last_fitness)
        distance_list.append(qpso.last_avg_distance)
        workload_var_list.append(qpso.last_workload_var)
        time.sleep(1)

    return fitness_list, distance_list, workload_var_list

def run_skenario6_comparison():
    max_iter = 220
    swarm_size = 20
    beta = 0.5
    mu = 0.3
    repeat = 5
    k_list = [20, 40, 60, 80, 100]

    results = []

    for k in k_list:
        for label, p_filter in [('0%', 0.0), ('50%', 0.5)]:
            logging.info(f"Running QPSO with PU filtering ({label}) for K={k}")
            
            if p_filter == 0.0:
                data = DataUtils('./dataset/bs_all.csv', './dataset/data_all.csv')
            else:
                data = DataUtilsPotential('./dataset/bs_all.csv', './dataset/data_all.csv', lower_percent=p_filter, upper_percent=1.0)

            bs_used = data.base_stations
            dist_used = data.distances

            if len(bs_used) == 0:
                logging.warning(f"Skipping PU {label} for K={k} because no base stations remain.")
                continue

            fitness_list, distance_list, workload_var_list = run_qpso_trial(
                bs_used, dist_used, k, max_iter, swarm_size, beta, mu, repeat)

            results.append({
                'filter': label,
                'num_edge_servers': k,
                **{f'Trial_{i+1}': f"{v:.5f}" for i, v in enumerate(fitness_list)},
                'Average': f"{np.mean(fitness_list):.5f}",
                'avg_distance': f"{np.mean(distance_list):.5f}",
                'workload_var': f"{np.mean(workload_var_list):.0f}"
            })

    df = pd.DataFrame(results)
    df.to_csv('results/csv/skenario6.csv', index=False)
    logging.info("✅ Saved to results/csv/skenario6.csv")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_skenario6_comparison()