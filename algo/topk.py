import logging
from datetime import datetime
from typing import List

from data.base_station import BaseStation
from data.edge_server import EdgeServer
from algo.server_placer import ServerPlacer


class TopKServerPlacer(ServerPlacer):
    """
    Top-K approach for edge server placement
    """
    name = 'TopK'

    def place_server(self, base_station_num, edge_server_num):
        logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                     f"Start running Top-K with N={base_station_num}, K={edge_server_num}")

        # Ambil base station sesuai jumlah input
        base_stations = self.base_stations[:base_station_num]

        # Urutkan base stations berdasarkan workload tertinggi
        sorted_base_stations = sorted(base_stations, key=lambda bs: bs.workload, reverse=True)

        # Tempatkan edge server pada lokasi base station dengan workload tertinggi
        edge_servers = [
            EdgeServer(i, bs.latitude, bs.longitude, bs.id)
            for i, bs in enumerate(sorted_base_stations[:edge_server_num])
        ]

        # Reset assignment
        for es in edge_servers:
            es.assigned_base_stations = []
            es.workload = 0

        # Alokasikan setiap base station ke edge server terdekat
        for bs in base_stations:
            closest_es = min(
                edge_servers,
                key=lambda es: self._distance_edge_server_base_station(es, bs)
            )
            closest_es.assigned_base_stations.append(bs)
            closest_es.workload += bs.workload

        self.edge_servers = [es for es in edge_servers if es.workload > 0]

        # Hitung dan simpan metrik evaluasi
        self.last_avg_distance = self.objective_avg_distance()
        self.last_workload_std = self.objective_workload_std()
        self.last_workload_var = self.objective_workload_variance()
        self.last_fitness = self.compute_combined_fitness()

        logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                     f"End running Top-K with fitness: {self.last_fitness:.5f}")
        logging.info(f"Final avg distance: {self.last_avg_distance:.5f}, workload var: {self.last_workload_var:.5f}")