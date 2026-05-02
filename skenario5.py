#SKENARIO 5 - METRIK JUMLAH SERVER
import time
import logging
import pandas as pd

from algo.qpso import QPSOServerPlacer
from utils.utils_all import DataUtils

def run_with_settings(placer, n, k, repeat_times=1):
    fitness_values = []
    for t in range(repeat_times):
        placer.place_server(n, k)
        fitness = placer.last_fitness
        fitness_values.append(fitness)
        time.sleep(1)
    return fitness_values

def run_qpso_server_count_experiment(data, j, max_iter=100, swarm_size=30, beta=0.75, mu=0.5):
    n = len(data.base_stations)
    k_range = list(range(20, 500, 20))

    records = []
    for k in k_range:
        logging.info(f"Running QPSO with server count K = {k}")
        qpso = QPSOServerPlacer(
            base_stations=data.base_stations,
            distances=data.distances,
            beta=beta,
            max_iter=max_iter,
            swarm_size=swarm_size,
            mu=mu
        )
        fitness_list = run_with_settings(qpso, n, k, repeat_times=1)
        row = {'K': k}
        for i, val in enumerate(fitness_list):
            row[f'Trial_{i+1}'] = val
        row['Average'] = sum(fitness_list) / len(fitness_list)
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(f'results/csv/skenario5_{j}.csv', index=False)
    logging.info(f"Saved to results/csv/skenario5_{j}.csv")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data = DataUtils('./dataset/bs_all.csv', './dataset/data_all.csv')
    for i in range(1, 11):
        logging.info(f"Running experiment {i}")
        run_qpso_server_count_experiment(data, i)