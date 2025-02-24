import numpy as np
from ocean.cascade import WavesCascade


class OceanWaves:

    SEA_STATE_BOUNDARIES = [(-1.0, 0.18), (-2.0, 0.19), (-5.0, 0.23),
                            (-9.0, 0.4), (-15.0, 0.6), (-20.0, 1.0),
                            (-47.0, 2.5), (-75.0, 3.0), (-125.0, 4.5),
                            (-175.0, 6.0), (-218.0, 9.0), (-241.0, 12.0),
                            (-255.0, 15.0)]

    def __init__(self, N, master_L, wind_speed, fetch, water_depth,
                 interpolation_degree):
        """
        Global ocean simulation which stores cascades and common simulation parameters.
        
        Parameters:
            N (int): Number of grid points on the master grid (and per cascade).
            master_L (float): Master domain length.
            wind_speed (float): Wind speed (m/s).
            fetch (float): Fetch distance (m).
            water_depth (float): Maximum depth (should be sufficiently large).
            interpolation_degree (int): Number of depth levels for velocity calculation.
        """
        self.N = N
        self.master_L = master_L
        self.wind_speed = wind_speed
        self.fetch = fetch
        self.water_depth = water_depth
        self.interpolation_degree = interpolation_degree

        # Compute sea state from wind speed
        self.sea_state = self.wind_speed_to_sea_state(wind_speed)

        # Get depth boundaries from sea state
        self.depth_boundaries = self.get_depth_boundaries(self.sea_state)

        # Master grid for merging cascade contributions.
        self.x = np.linspace(0, master_L, N)

        # Create cascades with default settings:
        self.cascades = []
        self.cascades.append(
            WavesCascade(N, 256, wind_speed, fetch, water_depth, 0,
                         (12 * np.pi) / 16, self.interpolation_degree))
        self.cascades.append(
            WavesCascade(N, 16, wind_speed, fetch, water_depth,
                         (12 * np.pi) / 16, (12 * np.pi) / 4,
                         self.interpolation_degree))
        self.cascades.append(
            WavesCascade(N, 4, wind_speed, fetch, water_depth,
                         (12 * np.pi) / 4, np.inf, self.interpolation_degree))

        # Logarithmic grid for velocity interpolation based on dynamic depth boundaries
        y_min, y_max = self.depth_boundaries
        self.velocity_depths = self.compute_log_grid(y_min, y_max,
                                                     interpolation_degree)

        # Immediately calculate the initial spectrum for each cascade.
        self.recalculate_initial_parameters()

    @staticmethod
    def wind_speed_to_sea_state(wind_speed):
        """
        Compute the sea state from wind speed.
        Formula: sea_state = 1.126 * wind_speed^(2/3)
        """
        return 1.126 * (wind_speed**(2 / 3))

    def get_depth_boundaries(self, sea_state):
        """
        Determine depth boundaries based on the computed sea state.
        
        Parameters:
            sea_state (float): The computed sea state.
        
        Returns:
            (float, float): The corresponding depth boundaries (y_min, y_max).
        """
        index = min(int(round(sea_state)), len(self.SEA_STATE_BOUNDARIES) - 1)
        return self.SEA_STATE_BOUNDARIES[index]

    def compute_log_grid(self, y_min, y_max, interpolation_degree):
        """
        Generate a logarithmic depth grid for velocity computation.

        Parameters:
            y_min (float): Minimum depth (negative, at ocean bottom).
            y_max (float): Maximum depth (0, at ocean surface).
            interpolation_degree (int): Number of depth levels.

        Returns:
            np.ndarray: Array of logarithmically spaced depth values.
        """
        grid_size = abs(y_max - y_min)
        interpolation_step = grid_size / (interpolation_degree - 2)

        beta = 1e-4
        alpha = -y_min / (2.0 * np.log(beta * abs(y_min)**2 + 1))

        grid = np.zeros(interpolation_degree)
        y = y_max
        grid[0] = y

        for i in range(1, interpolation_degree):
            sign = 1.0 if y >= 0 else -1.0
            grid[i] = sign * alpha * np.log(beta * abs(y)**2 + 1)
            y -= interpolation_step

        return grid

    def recalculate_initial_parameters(self):
        """
        (Initialization step) For each cascade, calculate h₀ and its conjugate.
        Mimics the CUDA function that initializes the spectrum.
        """
        for cascade in self.cascades:
            cascade.initialize_spectrum()

    def update_time_dependency(self, t):
        """
        Update the time-dependent part of the simulation in each cascade.
        """
        for cascade in self.cascades:
            cascade.update_time_dependency(t)

    def apply_ifft(self):
        """
        For each cascade, apply the inverse FFT to obtain real-space fields.
        """
        for cascade in self.cascades:
            cascade.apply_ifft()

    def update(self, t):
        """
        Update the ocean simulation in three parts (mimicking the CUDA code structure):
          1. Update the time dependency in each cascade.
          2. Apply the inverse FFT in each cascade.
        """
        self.update_time_dependency(t)
        self.apply_ifft()

    def get_real_water_height(self, X, N_iter=4):
        """
        Retrieve the "real" water height at the given world positions X.
        
        Inspired by your CUDA getWaterHeight kernel, the ocean surface is defined 
        parametrically as (x + D(x,t), h(x,t)). We iteratively correct the queried
        horizontal positions by subtracting the total horizontal displacement from all 
        cascades.
        
        Parameters:
            X (np.ndarray): Array of master grid positions (world coordinates).
            N_iter (int): Number of iterations.
            
        Returns:
            h_real (np.ndarray): The water height at positions X.
        """
        x_guess = X.copy()
        for _ in range(N_iter):
            total_disp = np.zeros_like(x_guess)
            for cascade in self.cascades:
                x_cascade = (x_guess / self.master_L) * cascade.L
                disp_cascade = np.interp(x_cascade, cascade.x,
                                         cascade.displacement)
                total_disp += disp_cascade
            x_guess = X - total_disp

        total_wh = np.zeros_like(x_guess)
        for cascade in self.cascades:
            x_cascade = (x_guess / self.master_L) * cascade.L
            wh_cascade = np.interp(x_cascade, cascade.x, cascade.water_height)
            total_wh += wh_cascade
        return total_wh
