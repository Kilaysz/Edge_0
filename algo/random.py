import random
import logging
from datetime import datetime

from algo.server_placer import ServerPlacer
from data.base_station import BaseStation
from data.edge_server import EdgeServer

class RandomServerPlacer(ServerPlacer):
    """
    Random approach for edge server placement
    """
    name = 'Random'

    def place_server(self, base_station_num, edge_server_num):
        logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                     f"Start running Random with N={base_station_num}, K={edge_server_num}")

        # Ambil base station yang akan diproses
        base_stations = self.base_stations[:base_station_num]

        # Pilih acak k lokasi sebagai lokasi edge server
        selected_bs = random.sample(base_stations, edge_server_num)

        # Buat edge server di lokasi base station terpilih
        edge_servers = [
            EdgeServer(i, bs.latitude, bs.longitude, bs.id)
            for i, bs in enumerate(selected_bs)
        ]

        # Reset assignment sebelum alokasi
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
                     f"End running Random with fitness: {self.last_fitness:.5f}")
        logging.info(f"Final avg distance: {self.last_avg_distance:.5f}, workload var: {self.last_workload_var:.5f}")