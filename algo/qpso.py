import logging
import random
import numpy as np
from datetime import datetime
from typing import List

from data.edge_server import EdgeServer
from data.base_station import BaseStation
from algo.server_placer import ServerPlacer

class QPSOServerPlacer(ServerPlacer): # extends ServerPlacer to implement the QPSO algorithm
    name = 'QPSO'

    def __init__(self, base_stations, distances: List[List[float]],  beta, max_iter, swarm_size, mu):
        super().__init__(base_stations)
        self.distances = distances
        self.beta = beta
        self.max_iter = max_iter
        self.swarm_size = swarm_size
        self.mu = mu

    # Implement the place_server method using the QPSO algorithm
    def place_server(self, base_station_num, edge_server_num):
        logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Start QPSO with N={base_station_num}, K={edge_server_num}, swarm_size={self.swarm_size}")

        n = base_station_num
        k = edge_server_num
        M = self.swarm_size

        swarm_pos = np.random.rand(M, n) # particle positions initialize randomly in [0, 1], where each position vector has a length equal to the number of base stations (n) and there are M particles in the swarm (X_i for i = 1 to M)
        pbest_pos = np.copy(swarm_pos) # personal best position for each particle (𝑝𝑏𝑒𝑠𝑡(0)= 𝐱(0))
        pbest_fitness = np.full(M, np.inf) # personal best fitness for each particle, initialized to infinity
        gbest_pos = None # initialize gbest to none
        gbest_fitness = np.inf # global best fitness initialized to infinity

        # Evaluate initial fitness for each particle and update pbest and gbest
        for i in range(M): # for swarm size 
            edge_servers = self.map_position_to_solution(swarm_pos[i], k) # map the particle position to a solution
            if not edge_servers:
                continue # if the solution is invalid (e.g. if k > number of base stations), skip it
            self.edge_servers = edge_servers # set the edge servers to the current solution to compute the fitness
            fit = self.compute_combined_fitness(self.mu) # compute the fitness of the current solution using the combined fitness function
            pbest_fitness[i] = fit
            if fit < gbest_fitness: # if the fitness of the current solution is better than the global best fitness, update the global best position and fitness
                gbest_fitness = fit
                gbest_pos = swarm_pos[i].copy()

        for iteration in range(self.max_iter):
            mbest = self.calculate_mbest(pbest_pos) # calculate the mean best position across all particles, which will be used in the position update formula to guide the particles towards the global best solution while maintaining diversity in the search process
            for i in range(M):
                new_pos = self.update_particle_position(swarm_pos[i], pbest_pos[i], gbest_pos, mbest) # yodare 
                swarm_pos[i] = new_pos

                edge_servers = self.map_position_to_solution(new_pos, k) # map the new particle position to a solution (i.e. a set of edge server placements) and evaluate its fitness, if the solution is invalid (e.g. if k > number of base stations), skip it and do not update pbest or gbest for this particle
                if not edge_servers:
                    continue
                self.edge_servers = edge_servers
                fit = self.compute_combined_fitness(self.mu) # compute the fitness of the new solution using the combined fitness function

                if fit < pbest_fitness[i]:
                    pbest_fitness[i] = fit
                    pbest_pos[i] = new_pos.copy()
                    if fit < gbest_fitness:
                        gbest_fitness = fit
                        gbest_pos = new_pos.copy()

            if iteration % 10 == 0 or iteration == self.max_iter - 1:
                logging.info(f"Iter {iteration + 1}/{self.max_iter} - Best fitness: {gbest_fitness:.5f}")

        final_servers = self.map_position_to_solution(gbest_pos, k)
        self.edge_servers = [es for es in final_servers if es.workload > 0] # filter out edge servers in final solution that have no workload assigned (only ones with workload > 0 are selected)
        self.last_fitness = gbest_fitness

        self.last_avg_distance = self.objective_avg_distance()
        self.last_workload_std = self.objective_workload_std()
        self.last_workload_var = self.objective_workload_variance()

        logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: End QPSO with best fitness {gbest_fitness:.5f}")
        logging.info(f"Final avg distance: {self.last_avg_distance:.5f}, workload var: {self.last_workload_var:.5f}")

    def calculate_mbest(self, pbest_pos):
        return np.mean(pbest_pos, axis=0) 

    # Update particle position based on QPSO formula
    def update_particle_position(self, x, p, g, mbest): # p = pbest, g = gbest, mbest = mean of pbest, x = current particle position 
        n = len(x) # the length of the position vector (number of base stations)
        u = np.random.rand(n) # random vector for stochasticity
        phi = np.random.rand(n) # random vector for stochasticity 
        direction = np.where(np.random.rand(n) < 0.5, 1, -1) # the sign of the update direction
        P = phi * p + (1 - phi) * g # local attractor 
        new_pos = P + direction * self.beta * np.abs(mbest - x) * np.log(1 / u) 
        return np.clip(new_pos, 0, 1) # bound it in [0, 1]

    # Calculate distance between edge server and base station using the pre-computed distance matrix (temporary solution until we implement the mapping of position to solution)
    def map_position_to_solution(self, pos_vector, k):
        sorted_idx = np.argsort(pos_vector)[::-1] # sort in descending order to get the indices of the top k values
        selected_locations = sorted_idx[:k] # select the top k indices.

        edge_servers = []
        for i, loc_idx in enumerate(selected_locations): # (i , loc_idx) = (edge server index, base station index)
            if loc_idx >= len(self.base_stations): # if the index exceeds the number of base stations, skip it (this can happen if k > number of base stations)
                continue
            bs = self.base_stations[loc_idx] # get the base station corresponding to the selected location index
            es = EdgeServer(i, bs.latitude, bs.longitude) # create an edge server at the location of the selected base station
            es.base_station_id = loc_idx # store the base station index for distance calculations later
            edge_servers.append(es) # append the edge server to the list of edge servers

        if not edge_servers: # if no edge servers were created (e.g. if k > number of base stations), return an empty list to indicate an invalid solution
            return []
        
        # initialize the edge servers
        for es in edge_servers:
            es.assigned_base_stations = []
            es.workload = 0

        # assign each base station to the nearest edge server based on the pre-computed distance matrix
        for bs_idx, bs in enumerate(self.base_stations):
            min_dist = float('inf')
            assigned_es = None
            # 
            for es in edge_servers: 
                dist = self._distance_edge_server_base_station(es, bs) # calculate distance using the pre-computed distance matrix if the edge server has a base station assigned, otherwise calculate it using the coordinates
                if dist < min_dist:
                    min_dist = dist
                    assigned_es = es
            if assigned_es:
                assigned_es.assigned_base_stations.append(bs)
                assigned_es.workload += bs.workload

        return edge_servers