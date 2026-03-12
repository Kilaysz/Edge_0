import logging
from datetime import datetime
from typing import List

import numpy as np
from sklearn.cluster import KMeans

from data.edge_server import EdgeServer
from data.base_station import BaseStation
from algo.server_placer import ServerPlacer

class KMeansServerPlacer(ServerPlacer):
    """
    K-means approach for edge server placement
    """
    name = 'KMeans'

    def place_server(self, base_station_num, edge_server_num):
        logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                     f"Start running K-Means with N={base_station_num}, K={edge_server_num}")

        # Gunakan base stations sesuai jumlah yang diberikan
        base_stations = self.base_stations[:base_station_num]
        coordinates = np.array([(bs.latitude, bs.longitude) for bs in base_stations])
        k = edge_server_num

        # Proses clustering K-Means
        kmeans = KMeans(n_clusters=k, random_state=0, max_iter=100, n_init='auto').fit(coordinates)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Buat edge server dan cari base station terdekat sebagai referensi
        edge_servers = []
        for i, (lat, lon) in enumerate(centroids):
            closest_bs = min(base_stations, key=lambda bs: (bs.latitude - lat)**2 + (bs.longitude - lon)**2)
            edge_server = EdgeServer(i, lat, lon, closest_bs.id)
            edge_servers.append(edge_server)

        # Reset assignment server
        for es in edge_servers:
            es.assigned_base_stations = []
            es.workload = 0

        # Alokasikan base station ke cluster terdekat
        for idx, cluster_id in enumerate(labels):
            bs = base_stations[idx]
            es = edge_servers[cluster_id]
            es.assigned_base_stations.append(bs)
            es.workload += bs.workload

        # Simpan server yang aktif saja
        self.edge_servers = [es for es in edge_servers if es.workload > 0]

        # Hitung dan simpan semua metrik evaluasi
        self.last_avg_distance = self.objective_avg_distance()
        self.last_workload_std = self.objective_workload_std()
        self.last_workload_var = self.objective_workload_variance()
        self.last_fitness = self.compute_combined_fitness()

        logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                     f"End running K-Means with fitness: {self.last_fitness:.5f}")
        logging.info(f"Final avg distance: {self.last_avg_distance:.5f}, workload var: {self.last_workload_var:.5f}")