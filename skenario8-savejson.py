# SKENARIO 8 - PERBANDINGAN METODE - FILTERING PU 50% + Simpan Lokasi Server
import time
import logging
import pandas as pd
import numpy as np
import json

from algo.qpso import QPSOServerPlacer
from algo.kmeans import KMeansServerPlacer
from algo.topk import TopKServerPlacer
from algo.random import RandomServerPlacer
from algo.weighted_kmeans import WeightedKMeansServerPlacer

from utils.utils_all import DataUtils

def compute_pu_and_filter(base_stations, lower_percent=0.5, upper_percent=1.0):
    workloads = np.array([bs.workload for bs in base_stations])
    visits = np.array([bs.num_users for bs in base_stations])

    norm_workload = (workloads - workloads.min()) / (workloads.max() - workloads.min() + 1e-9)
    norm_visits = (visits - visits.min()) / (visits.max() - visits.min() + 1e-9)

    for i, bs in enumerate(base_stations):
        bs.pu = norm_workload[i] + norm_visits[i]

    sorted_bs = sorted(base_stations, key=lambda x: x.pu, reverse=True)
    total = len(sorted_bs)
    lower_idx = int(total * lower_percent)
    upper_idx = int(total * upper_percent)
    filtered_bs = sorted_bs[lower_idx:upper_idx]

    for new_id, bs in enumerate(filtered_bs):
        bs.id = new_id

    distances = []
    for bs1 in filtered_bs:
        row = [DataUtils.calc_distance(bs1.latitude, bs1.longitude, bs2.latitude, bs2.longitude) for bs2 in filtered_bs]
        distances.append(row)

    logging.info(f"{len(filtered_bs)} base stations selected after PU filtering (50%)")
    return filtered_bs, distances

def compute_total_distance_and_workload(placer):
    total_distance = 0.0
    total_workload = 0.0

    for es in placer.edge_servers:
        for bs in es.assigned_base_stations:
            if es.base_station_id is not None:
                dist = placer.distances[es.base_station_id][bs.id]
            else:
                dist = placer._distance_edge_server_base_station(es, bs)
            total_distance += dist
            total_workload += bs.workload

    return total_distance, total_workload

def extract_server_locations(placer):
    server_data = []
    for es in placer.edge_servers:
        if es.base_station_id is None:
            continue
        bs_server = placer.base_stations[es.base_station_id]
        server_info = {
            'latitude': bs_server.latitude,
            'longitude': bs_server.longitude,
            'assigned_bs': []
        }
        for bs in es.assigned_base_stations:
            server_info['assigned_bs'].append({
                'latitude': bs.latitude,
                'longitude': bs.longitude
            })
        server_data.append(server_info)
    return server_data

def run_with_timer(placer, n, k, name):
    start = time.time()
    placer.place_server(n, k)
    end = time.time()

    result = {
        'avg_distance': getattr(placer, 'last_avg_distance', None),
        'workload_variance': getattr(placer, 'last_workload_var', None),
        'workload_std': getattr(placer, 'last_workload_std', None),
        'fitness': getattr(placer, 'last_fitness', None),
        'time': round(end - start, 4),
    }

    total_dist, total_work = compute_total_distance_and_workload(placer)
    result['total_distance'] = total_dist
    result['total_workload'] = total_work

    # Simpan lokasi server ke file JSON
    locations = extract_server_locations(placer)
    filename = f'results/json/skenario8_{name.lower().replace("-", "").replace(" ", "")}_k{k}_locations.json'
    with open(filename, 'w') as f:
        json.dump(locations, f, indent=2)

    return result

def run_skenario8_with_filtering():
    logging.basicConfig(level=logging.INFO)

    data = DataUtils('./dataset/bs_all.csv', './dataset/data_all.csv')

    filtered_bs, new_distances = compute_pu_and_filter(data.base_stations, lower_percent=0.5, upper_percent=1.0)
    n = len(filtered_bs)

    qpso_params = {
        'max_iter': 240,
        'swarm_size': 10,
        'beta': 0.6,
        'mu': 0.3
    }

    k_values = list(range(20, 101, 20))
    records = []

    for k in k_values:
        print(f"\nEvaluating for K = {k} edge servers (Filtered 50%)")

        placers = {
            'QPSO': QPSOServerPlacer(filtered_bs, new_distances, **qpso_params),
            'K-means': KMeansServerPlacer(filtered_bs, new_distances),
            'Top-K': TopKServerPlacer(filtered_bs, new_distances),
            'Random': RandomServerPlacer(filtered_bs, new_distances),
            'WeightedKMeans': WeightedKMeansServerPlacer(filtered_bs, new_distances),
        }

        for name, placer in placers.items():
            print(f"  -> Running {name}")
            logging.info(f"Running {name} with K = {k}")
            result = run_with_timer(placer, n, k, name)
            result.update({
                'placer_name': name,
                'num_base_stations': n,
                'num_edge_servers': k,
                'filtering': '50%'
            })
            records.append(result)

    df = pd.DataFrame(records)
    df.to_csv('results/csv/skenario8.csv', index=False)
    logging.info("✅ Saved to results/csv/skenario8.csv")

if __name__ == '__main__':
    run_skenario8_with_filtering()