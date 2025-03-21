class OceanInitConfig:

    def __init__(
        self, N, master_L, wind_speed, fetch, water_depth, interpolation_degree
    ):
        self.N = N
        self.master_L = master_L
        self.wind_speed = wind_speed
        self.fetch = fetch
        self.water_depth = water_depth
        self.interpolation_degree = interpolation_degree