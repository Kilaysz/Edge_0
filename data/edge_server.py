class EdgeServer(object):
    
    def __init__(self, id, latitude, longitude, base_station_id=None):
        self.id = id
        self.latitude = latitude
        self.longitude = longitude
        self.base_station_id = base_station_id
        self.assigned_base_stations = [] # List of base station IDs assigned to this edge server
        self.workload = 0 # How much workload this edge server has (total used time in minutes)