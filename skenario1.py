#SKENARIO 1 - JUMLAH ITERASI 
import time
import logging
import pandas as pd

from algo.qpso import QPSOServerPlacer
from utils.utils_all import DataUtils

def run_with_settings(placer, n, k, repeat_times=5):
    fitness_values = []
    for t in range(repeat_times):
        placer.place_server(n, k) # run the server placement algo
        fitness = placer.last_fitness # get the last fitness value from the placer
        fitness_values.append(fitness) # append to the list of fitness values
        time.sleep(1)
    return fitness_values # return the list of fitness values for this setting

# Eksperimen untuk variasi jumlah iterasi
def run_qpso_iter_experiment(data, j, k=100, swarm_size=30, beta=0.75, mu=0.5):
    n = len(data.base_stations)
    iter_range = list(range(20, 301, 20))
    records = []

    for max_iter in iter_range:
        logging.info(f"Running QPSO with max_iter = {max_iter}")
        qpso = QPSOServerPlacer(
            base_stations=data.base_stations,
            distances=data.distances,
            beta=beta,
            max_iter=max_iter,
            swarm_size=swarm_size,
            mu=mu
        ) # initialize the QPSO server placer with the current max_iter
        fitness_list = run_with_settings(qpso, n, k, repeat_times=5) # run the experiment and get fitness values  
        row = {'max_iter': max_iter} # iteration count for this row
        for i, val in enumerate(fitness_list): 
            row[f'Trial_{i+1}'] = val
        row['Average'] = sum(fitness_list) / len(fitness_list)
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(f'results/csv/skenario1_{j}.csv', index=False)
    logging.info(f"Saved to results/csv/skenario1_{j}.csv") 

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data = DataUtils('./dataset/bs_all.csv', './dataset/data_all.csv')
    for i in range(1, 11):
        logging.info(f"Running experiment {i}")
        run_qpso_iter_experiment(data, i)