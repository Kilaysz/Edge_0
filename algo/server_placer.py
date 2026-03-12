import logging
from typing import List
import numpy as np

from data.base_station import BaseStation
from data.edge_server import EdgeServer
from utils.utils_all import DataUtils
from utils.utils_potential import DataUtilsPotential

class ServerPlacer(object): # provide a blueprint for qpso to inherit.
    def __init__(self, base_stations: List[BaseStation]):
        self.base_stations = base_stations.copy() # create a copy to avoid modifying the original list
        self.edge_servers = None # will be set by the specific algorithm implementation

        # Deteksi apakah base station punya atribut PU
        has_pu = hasattr(self.base_stations[0], 'pu')
        self.data_utils = DataUtilsPotential if has_pu else DataUtils # if it has pu, assign it to data utils potential, else data utils

        # Perhitungan jarak otomatis
        self.distances = self.data_utils.distance_between_stations(self.base_stations)

        self.last_fitness = None
        self.Damax = self.estimate_max_avg_distance()
        self.Bwmax = self.estimate_max_workload_var()

    def place_server(self, base_station_num, edge_server_num): # need a method to implement this function
        raise NotImplementedError

    def _distance_edge_server_base_station(self, edge_server: EdgeServer, base_station: BaseStation) -> float:
        try:
            if edge_server.base_station_id is not None:
                return self.distances[edge_server.base_station_id][base_station.id]
        except IndexError:
            pass
        return self.data_utils.calc_distance(
            edge_server.latitude, edge_server.longitude,
            base_station.latitude, base_station.longitude
        )

    def compute_objectives(self):
        avg_distance = self.objective_avg_distance()
        workload_std = self.objective_workload_std()
        workload_var = self.objective_workload_variance()

        self.last_avg_distance = avg_distance
        self.last_workload_std = workload_std
        self.last_workload_var = workload_var

        return {
            'avg_distance': avg_distance,
            'workload_std': workload_std,
            'workload_var': workload_var
        }

    def objective_avg_distance(self):
        total_distance = 0
        bs_count = 0
        for es in self.edge_servers:
            for bs in es.assigned_base_stations:
                dist = self._distance_edge_server_base_station(es, bs)
                total_distance += dist
                bs_count += 1
        return total_distance / bs_count if bs_count else float('inf')

    def objective_workload_std(self):
        workloads = [es.workload for es in self.edge_servers]
        return np.std(workloads) if workloads else float('inf')

    def objective_workload_variance(self):
        workloads = [es.workload for es in self.edge_servers]
        return np.var(workloads) if workloads else float('inf')

    def estimate_max_avg_distance(self):
        all_dists = [d for row in self.distances for d in row]
        return max(all_dists) if all_dists else 1

    def estimate_max_workload_std(self):
        workloads = [bs.workload for bs in self.base_stations]
        total = sum(workloads)
        return np.std([total, 0])

    def estimate_max_workload_var(self):
        workloads = [bs.workload for bs in self.base_stations]
        total = sum(workloads)
        return np.var([total, 0])

    def compute_combined_fitness(self, mu=0.5): # fitness function that combines the average distance
        avg_distance = self.objective_avg_distance()
        workload_var = self.objective_workload_variance()

        norm_distance = avg_distance / self.Damax if self.Damax else float('inf') # normalize the average distance by the estimated maximum average distance to get a value 
        norm_workload = workload_var / self.Bwmax if self.Bwmax else float('inf') # normalize the workload variance by the estimated maximum workload variance to get a value between 0 and 1

        fitness = mu * norm_distance + (1 - mu) * norm_workload
        self.last_fitness = fitness 
        return fitness # dimension = 1, a single value representing the combined fitness of the current solution, where lower is better.