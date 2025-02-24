import numpy as np
from ocean.cascade import WavesCascade


class OceanWaves:

    def __init__(self, N, master_L, wind_speed, fetch, water_depth):
        """
        Global ocean simulation which stores cascades and common simulation parameters.
        
        Parameters:
            N (int): Number of grid points on the master grid (and per cascade).
            master_L (float): Master domain length.
            wind_speed, fetch, water_depth: Common ocean parameters.
        """
        self.N = N
        self.master_L = master_L
        self.wind_speed = wind_speed
        self.fetch = fetch
        self.water_depth = water_depth

        # Master grid for merging cascade contributions.
        self.x = np.linspace(0, master_L, N)

        # Create cascades with default settings:
        self.cascades = []
        # Cascade 0: L0 = 256, operating for |k| in [0, 12π/16)
        self.cascades.append(
            WavesCascade(N, 256, wind_speed, fetch, water_depth, 0,
                         (12 * np.pi) / 16))
        # Cascade 1: L1 = 16, operating for |k| in [12π/16, 12π/4)
        self.cascades.append(
            WavesCascade(N, 16, wind_speed, fetch, water_depth,
                         (12 * np.pi) / 16, (12 * np.pi) / 4))
        # Cascade 2: L2 = 4, operating for |k| in [12π/4, ∞)
        self.cascades.append(
            WavesCascade(N, 4, wind_speed, fetch, water_depth,
                         (12 * np.pi) / 4, np.inf))

        # Immediately calculate the initial spectrum for each cascade.
        self.recalculate_initial_parameters()

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
          3. (Merging is done later in get_real_water_height.)
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
            # For each cascade, re-map x_guess to the cascade coordinate and interpolate its displacement.
            for cascade in self.cascades:
                x_cascade = (x_guess / self.master_L) * cascade.L
                disp_cascade = np.interp(x_cascade, cascade.x,
                                         cascade.displacement)
                total_disp += disp_cascade
            x_guess = X - total_disp
        # Once converged, sum the water height contributions from each cascade.
        total_wh = np.zeros_like(x_guess)
        for cascade in self.cascades:
            x_cascade = (x_guess / self.master_L) * cascade.L
            wh_cascade = np.interp(x_cascade, cascade.x, cascade.water_height)
            total_wh += wh_cascade
        return total_wh
