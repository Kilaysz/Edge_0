import random
import csv
import logging
import os
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
from functools import wraps
from math import cos, asin, sqrt
from typing import List

from data.base_station import BaseStation

def memorize(filename):
    def _memorize(func):
        @wraps(func)
        def memorized_function(*args, **kwargs):
            key = pickle.dumps(args[1:])
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    cached = pickle.load(f)
                    f.close()
                    if isinstance(cached, dict) and cached.get('key') == key:
                        logging.info(f'Found cache: {filename}, {func.__name__} does not need to run')
                        return cached['value']
            value = func(*args, **kwargs)
            with open(filename, 'wb') as f:
                cached = {'key': key, 'value': value}
                pickle.dump(cached, f)
                f.close()
            return value
        return memorized_function
    return _memorize

class DataUtilsPotential(object):
    def __init__(self, location_file, user_info_file):
        self.all_base_stations = self.base_station_reader(location_file)
        self.base_stations = self.user_info_reader(user_info_file)
        self.filter_top_pu_base_stations()  # hanya 50% atas yang dipilih
        self.distances = self.distance_between_stations() # calculate distance between base stations, this will be used by server placer to calculate distance between edge server and base station

    @memorize('cache_potential/base_stations')
    def base_station_reader(self, path: str):
        bs_data = pd.read_csv(path, header=0, index_col=0)
        base_stations = []
        for index, bs_info in bs_data.iterrows():
            base_stations.append(BaseStation(id=index, addr=bs_info['address'], lat=bs_info['latitude'], lng=bs_info['longitude']))
            logging.debug(f"(Base station: {index}: address={bs_info['address']}, latitude={bs_info['latitude']}, longitude={bs_info['longitude']})")
        return base_stations

    @memorize('cache_potential/base_stations_with_user_info')
    def user_info_reader(self, path: str) -> List[BaseStation]:
        self.address_to_id = {bs.address: bs.id for bs in self.all_base_stations}
        req_data = pd.read_csv(path, header=0, index_col=0)
        req_data['start time'] = pd.to_datetime(req_data['start time'])
        req_data['end time'] = pd.to_datetime(req_data['end time'])

        for index, req_info in req_data.iterrows():
            service_time = (req_info['end time'] - req_info['start time']).seconds / 60
            bs_id = self.address_to_id[req_info['address']]
            self.all_base_stations[bs_id].num_users += 1
            self.all_base_stations[bs_id].workload += service_time
            logging.debug(f"(User info: address={req_info['address']}, begin_time={req_info['start time']}, end_time={req_info['end time']})")

        return self.all_base_stations

    def filter_top_pu_base_stations(self, percentage=0.5):
        # Hitung PU untuk semua base station
        workloads = np.array([bs.workload for bs in self.all_base_stations])
        visits = np.array([bs.num_users for bs in self.all_base_stations])


        norm_workload = (workloads - workloads.min()) / (workloads.max() - workloads.min() + 1e-9)
        norm_visits = (visits - visits.min()) / (visits.max() - visits.min() + 1e-9)

        for i, bs in enumerate(self.all_base_stations):
            # pu = normalized workload + normalized visits
            bs.pu = norm_workload[i] + norm_visits[i]

        # sort the base stations by PU and select the top percentage
        sorted_bs = sorted(self.all_base_stations, key=lambda x: x.pu, reverse=True)
        count = int(len(sorted_bs) * percentage)
        # pick the top percentage of base stations
        self.base_stations = sorted_bs[:count]

        # Re-assign ID untuk keperluan indexing
        for new_id, bs in enumerate(self.base_stations):
            bs.id = new_id

        logging.info(f"{len(self.base_stations)} base stations selected after PU filtering (Top {int(percentage * 100)}%)")

    @staticmethod
    def _shuffle(l: List):
        random.seed(6767)
        random.shuffle(l)

    @staticmethod
    def calc_distance(lat_a, lng_a, lat_b, lng_b):
        p = 0.017453292519943295
        a = 0.5 - cos((lat_b - lat_a) * p) / 2 + cos(lat_a * p) * cos(lat_b * p) * (1 - cos((lng_b - lng_a) * p)) / 2
        return 12742 * asin(sqrt(a))

    @memorize('cache_potential/distances')
    def distance_between_stations(self) -> List[List[float]]:
        assert self.base_stations
        distances = []
        for i, station_a in enumerate(self.base_stations):
            distances.append([])
            for j, station_b in enumerate(self.base_stations):
                dist = DataUtilsPotential.calc_distance(station_a.latitude, station_a.longitude,
                                               station_b.latitude, station_b.longitude)
                distances[i].append(dist)
            logging.debug(f"Calculated distance from base station {i} to others.")
        return distances