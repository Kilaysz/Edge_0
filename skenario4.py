#SKENARIO 4 - NILAI MU
import time
import logging
import pandas as pd
import numpy as np

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

def run_qpso_mu_experiment(data, k=100, max_iter=100, swarm_size=30, beta=0.75):
    n = len(data.base_stations)
    mu_range = [round(m, 2) for m in np.arange(0.1, 1.1, 0.1)]

    records = []
    for mu in mu_range:
        logging.info(f"Running QPSO with mu = {mu}")
        qpso = QPSOServerPlacer(
            base_stations=data.base_stations,
            distances=data.distances,
            beta=beta,
            max_iter=max_iter,
            swarm_size=swarm_size,
            mu=mu
        )
        fitness_list = run_with_settings(qpso, n, k, repeat_times=5)
        row = {'mu': mu}
        for i, val in enumerate(fitness_list):
            row[f'Trial_{i+1}'] = val
        row['Average'] = sum(fitness_list) / len(fitness_list)
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv('results/csv/skenario4.csv', index=False)
    logging.info("Saved to results/csv/skenario4.csv")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data = DataUtils('./dataset/bs_all.csv', './dataset/data_all.csv')
    run_qpso_mu_experiment(data)