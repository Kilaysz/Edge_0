#SKENARIO 3 - NILAI BETA
import time
import logging
import pandas as pd

from algo.qpso import QPSOServerPlacer
from utils.utils_all import DataUtils

def run_with_settings(placer, n, k, repeat_times=5):
    fitness_values = []
    for t in range(repeat_times):
        placer.place_server(n, k)
        fitness = placer.last_fitness
        fitness_values.append(fitness)
        time.sleep(1)
    return fitness_values

def run_qpso_beta_experiment(data, k=100, max_iter=100, swarm_size=30, mu=0.5):
    n = len(data.base_stations)
    beta_range = [round(b, 2) for b in list(np.arange(1.0, 1.1, 0.1))]

    records = []
    for beta in beta_range:
        logging.info(f"Running QPSO with beta = {beta}")
        qpso = QPSOServerPlacer(
            base_stations=data.base_stations,
            distances=data.distances,
            beta=beta,
            max_iter=max_iter,
            swarm_size=swarm_size,
            mu=mu
        )
        fitness_list = run_with_settings(qpso, n, k, repeat_times=5)
        row = {'beta': beta}
        for i, val in enumerate(fitness_list):
            row[f'Trial_{i+1}'] = val
        row['Average'] = sum(fitness_list) / len(fitness_list)
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv('results/csv/skenario3_1.csv', index=False)
    logging.info("Saved to results/csv/skenario3.csv")

if __name__ == '__main__':
    import numpy as np  
    logging.basicConfig(level=logging.INFO)
    data = DataUtils('./dataset/bs_all.csv', './dataset/data_all.csv')
    run_qpso_beta_experiment(data)