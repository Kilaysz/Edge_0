# SKENARIO 7 - PERBANDINGAN METODE - NON PU + Simpan Lokasi Server per Algoritma & K
#
# File runs a comparison of different edge-server placement algorithms using
# the full dataset (non-PU filtered). For each algorithm and each K (number of
# edge servers) it records performance metrics and also saves the geographical
# locations of the chosen servers along with the base stations assigned to
# them. The results are written to CSV and JSON files under the results/ folder.

import time
import logging
import pandas as pd
import json

from algo.qpso import QPSOServerPlacer
from algo.kmeans import KMeansServerPlacer
from algo.topk import TopKServerPlacer
from algo.random import RandomServerPlacer
from algo.weighted_kmeans import WeightedKMeansServerPlacer
from utils.utils_all import DataUtils


def compute_total_distance_and_workload(placer):
    # Walk through every edge server configured by the placer and accumulate
    # two aggregate metrics:
    #  * total_distance: the sum of distances from each assigned BS to its
    #    serving edge server (uses precomputed distance matrix when possible).
    #  * total_workload: the sum of workloads of all BSs that have been
    #    assigned. This is useful for verifying that total demand is constant.
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
    # Build a JSON-serializable structure describing where each edge server
    # ended up (latitude/longitude) and which base stations were assigned to
    # it. Used later for visualisation or mapping of algorithm results.
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
    # Execute the placer.place_server() call and measure elapsed time. After
    # placement we gather a small set of diagnostic metrics that are exposed
    # by the placer implementations (avg distance, workload variance etc.)
    # and also compute total distance/workload by calling helper functions.
    #
    # Additionally, the chosen server locations are exported into a JSON file
    # named according to the algorithm and K value so that they can later be
    # plotted or inspected.
    start = time.time()
    placer.place_server(n, k)
    end = time.time()

    result = {
        'avg_distance': getattr(placer, 'last_avg_distance', None),
        'workload_var': getattr(placer, 'last_workload_var', None),
        'workload_std': getattr(placer, 'last_workload_std', None),
        'fitness': getattr(placer, 'last_fitness', None),
        'time': round(end - start, 4),
    }

    total_dist, total_work = compute_total_distance_and_workload(placer)
    result['total_distance'] = total_dist
    result['total_workload'] = total_work

    # Simpan file JSON lokasi server untuk setiap algoritma dan K
    # (format: results/json/main_<algoname>_k<k>_locations.json)
    locations = extract_server_locations(placer)
    filename = f'results/json/main_{name.lower().replace("-", "").replace(" ", "")}_k{k}_locations.json'
    with open(filename, 'w') as f:
        json.dump(locations, f, indent=2)

    return result


def run_skenario7_comparison():
    # Driver function that sets up the dataset (non-PU case) and iterates
    # through a series of K values. For each K it instantiates one placer
    # object per algorithm, runs them, times them and appends the results into
    # a list of records which is finally saved as a CSV.
    logging.basicConfig(level=logging.INFO)

    # Non-PU: gunakan data tanpa filtering PU. This loads all base stations
    # and computes the full distance matrix.
    data = DataUtils('./dataset/bs_all.csv', './dataset/data_all.csv')
    n = len(data.base_stations)

    # Parameters specific to the QPSO algorithm; the others don't require
    # additional tuning parameters here.
    qpso_params = {
        'max_iter': 220,
        'swarm_size': 20,
        'beta': 0.5,
        'mu': 0.3
    }

    # Test K at 20,40,60,80,100 edge servers
    k_values = list(range(20, 101, 20))
    records = []

    for k in k_values:
        print(f"\nEvaluating for K = {k} edge servers")

        # Instantiate one placer object per algorithm. They each take the
        # same list of base stations and distance matrix so the comparison
        # is fair.
        placers = {
            'QPSO': QPSOServerPlacer(data.base_stations, data.distances, **qpso_params),
            'K-means': KMeansServerPlacer(data.base_stations, data.distances),
            'Top-K': TopKServerPlacer(data.base_stations, data.distances),
            'Random': RandomServerPlacer(data.base_stations, data.distances),
            'WeightedKMeans': WeightedKMeansServerPlacer(data.base_stations, data.distances),
        }

        for name, placer in placers.items():
            print(f"  -> Running {name}")
            logging.info(f"Running {name} with K = {k}")
            result = run_with_timer(placer, n, k, name) # run the placer and get metrics
            result.update({
                'placer_name': name,
                'num_base_stations': n,
                'num_edge_servers': k
            })
            records.append(result)

    # Convert the list of result dictionaries to a DataFrame and persist
    # the aggregated metrics to CSV for later analysis.
    df = pd.DataFrame(records)
    df.to_csv('results/csv/skenario7.csv', index=False)
    logging.info("✅ Saved to results/csv/skenario7.csv")


if __name__ == '__main__':
    run_skenario7_comparison()