import numpy as np
from ocean.cascade import WavesCascade
from ocean.init_helper import OceanInitConfig


class OceanWaves:
    SEA_STATE_BOUNDARIES = [
        (-1.0, 0.18),
        (-2.0, 0.19),
        (-5.0, 0.23),
        (-9.0, 0.4),
        (-15.0, 0.6),
        (-20.0, 1.0),
        (-47.0, 2.5),
        (-75.0, 3.0),
        (-125.0, 4.5),
        (-175.0, 6.0),
        (-218.0, 9.0),
        (-241.0, 12.0),
        (-255.0, 15.0),
    ]

    def __init__(self, config):
        """
        Global ocean simulation which stores cascades and common simulation
        parameters.
        """
        self.N = config.N
        self.master_L = config.master_L
        self.wind_speed = config.wind_speed
        self.fetch = config.fetch
        self.water_depth = config.water_depth
        self.interpolation_degree = config.interpolation_degree

        # Compute sea state from wind speed.
        self.sea_state = self.wind_speed_to_sea_state(self.wind_speed)

        # Get depth boundaries from sea state.
        self.depth_boundaries = self.get_depth_boundaries(self.sea_state)
        # Logarithmic grid for velocity interpolation based on dynamic depth
        # boundaries.
        y_min, y_max = self.depth_boundaries
        self.velocity_depths = self.compute_log_grid(
            y_min, y_max, self.interpolation_degree
        )

        # Master grid for merging cascade contributions.
        self.x = np.linspace(0, self.master_L, self.N)

        # Define arrays for cascade parameters
        cascade_n_values = [256, 16, 4]
        cascade_freq_starts = [0, (12 * np.pi) / 16, (12 * np.pi) / 4]
        cascade_freq_ends = [(12 * np.pi) / 16, (12 * np.pi) / 4, np.inf]

        # Create cascades with default settings.
        self.cascades = [
            WavesCascade(
                self.N,
                cascade_n_values[0],
                self.wind_speed,
                self.fetch,
                self.water_depth,
                cascade_freq_starts[0],
                cascade_freq_ends[0],
                self.interpolation_degree,
                self.velocity_depths,
            ),
            WavesCascade(
                self.N,
                cascade_n_values[1],
                self.wind_speed,
                self.fetch,
                self.water_depth,
                cascade_freq_starts[1],
                cascade_freq_ends[1],
                self.interpolation_degree,
                self.velocity_depths,
            ),
            WavesCascade(
                self.N,
                cascade_n_values[2],
                self.wind_speed,
                self.fetch,
                self.water_depth,
                cascade_freq_starts[2],
                cascade_freq_ends[2],
                self.interpolation_degree,
                self.velocity_depths,
            ),
        ]

        # Immediately calculate the initial spectrum for each cascade.
        self.recalculate_initial_parameters()

    @staticmethod
    def wind_speed_to_sea_state(wind_speed):
        """
        Compute the sea state from wind speed.
        Formula: sea_state = 1.126 * wind_speed^(2/3)
        """
        return 1.126 * (wind_speed ** (2 / 3))

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
        alpha = -y_min / (2.0 * np.log(beta * abs(y_min) ** 2 + 1))

        grid = np.zeros(interpolation_degree)
        y = y_max
        grid[0] = y

        for i in range(1, interpolation_degree):
            sign = 1.0 if y >= 0 else -1.0
            grid[i] = sign * alpha * np.log(beta * abs(y) ** 2 + 1)
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
        Update the ocean simulation in three parts (mimicking the CUDA code
          structure): 1. Update the time dependency in each cascade.
          2. Apply the inverse FFT in each cascade.

        """
        self.update_time_dependency(t)
        self.apply_ifft()

    def get_real_water_height(self, X, N_iter=4):
        # Extract cascade attributes.
        L_list = [cascade.L for cascade in self.cascades]
        x_list = [cascade.x for cascade in self.cascades]
        displacement_list = [cascade.displacement for cascade in self.cascades]
        water_height_list = [cascade.water_height for cascade in self.cascades]
        return static_get_real_water_height(
            X,
            self.master_L,
            L_list,
            x_list,
            displacement_list,
            water_height_list,
            N_iter,
        )

    def get_real_water_velocity(self, X, Y):
        L_list = [cascade.L for cascade in self.cascades]
        x_list = [cascade.x for cascade in self.cascades]
        displacement_list = [cascade.displacement for cascade in self.cascades]
        water_height_list = [cascade.water_height for cascade in self.cascades]
        velocity_list = [cascade.velocity for cascade in self.cascades]
        return static_get_real_water_velocity(
            X,
            Y,
            self.master_L,
            L_list,
            x_list,
            displacement_list,
            water_height_list,
            velocity_list,
            self.velocity_depths,
            self.interpolation_degree,
        )


def static_get_real_water_height(
    X, master_L, L_list, x_list, displacement_list, water_height_list, N_iter=4
):
    """
        Compute the "real" water height at positions X with periodic wrapping.

        Parameters:
            X (np.ndarray): 1D array of master grid positions.
            master_L (float): The master domain length.
            L_list (list of float): List of cascade domain lengths.
            x_list (list of np.ndarray): List of 1D arrays with cascade grid
            positions. displacement_list (list of np.ndarray): List of 1D
            arrays with cascade displacements. water_height_list (list of
            np.ndarray): List of 1D arrays with cascade water heights. N_iter
    (int): Number of correction iterations (default: 4).
        Returns:
            np.ndarray: The computed water height at each position in X.

    
    """
    # Wrap X to ensure periodic tiling.
    X_mod = np.mod(X, master_L)
    x_guess = X_mod.copy()
    for _ in range(N_iter):
        total_disp = np.zeros_like(x_guess)
        for L, x_grid, disp in zip(L_list, x_list, displacement_list):
            # Map the periodic master grid positions into cascade coordinates.
            x_cascade = (x_guess / master_L) * L
            total_disp += np.interp(x_cascade, x_grid, disp)
        # Subtract displacement and re-wrap to keep periodicity.
        x_guess = np.mod(X - total_disp, master_L)

    total_wh = np.zeros_like(x_guess)
    for L, x_grid, wh in zip(L_list, x_list, water_height_list):
        x_cascade = (x_guess / master_L) * L
        total_wh += np.interp(x_cascade, x_grid, wh)
    return total_wh


def static_get_real_water_velocity(
    X,
    Y,
    master_L,
    L_list,
    x_list,
    displacement_list,
    water_height_list,
    velocity_list,
    velocity_depths,
    interpolation_degree,
):
    """
        Compute the water velocity at given world positions (X, Y) with
    periodic tiling.
        Parameters:
            X (np.ndarray): 1D array of horizontal master grid positions.
            Y (np.ndarray): 1D array of vertical positions (depths).
            master_L (float): The master domain length.
            L_list (list of float): List of cascade domain lengths.
            x_list (list of np.ndarray): List of 1D arrays with cascade grid
            positions. displacement_list (list of np.ndarray): List of 1D
            arrays with cascade displacements. water_height_list (list of
            np.ndarray): List of 1D arrays with cascade water heights.
                                                velocity_list (list of
            np.ndarray): List of 3D arrays with cascade velocity fields (each
            of shape (interpolation_degree, N, 2)). velocity_depths
    (np.ndarray): 1D array of logarithmically spaced depth values.
        interpolation_degree (int): Number of depth slices. Returns:
            np.ndarray: Array of shape (len(X), 2) containing (vx, vy) for each
        query position.
    
    """
    # First, compute the water height using the periodic height function.
    h_vals = static_get_real_water_height(
        X,
        master_L,
        L_list,
        x_list,
        displacement_list,
        water_height_list,
        N_iter=4,
    )

    velocities = np.zeros((len(X), 2))
    for idx, (x, y) in enumerate(zip(X, Y)):
        # If above the water surface, velocity remains zero.
        if y > h_vals[idx]:
            continue

        velocity_slices = np.zeros((interpolation_degree, 2))
        for L, x_grid, vel_field in zip(L_list, x_list, velocity_list):
            # Wrap the horizontal coordinate periodically before scaling.
            x_scaled = ((x % master_L) / master_L) * L
            for i in range(interpolation_degree):
                v_interp0 = np.interp(x_scaled, x_grid, vel_field[i, :, 0])
                v_interp1 = np.interp(x_scaled, x_grid, vel_field[i, :, 1])
                velocity_slices[i] += np.array([v_interp0, v_interp1])

        # Perform exponential interpolation along depth.
        if y <= velocity_depths[0]:
            velocities[idx] = velocity_slices[0]
        elif y >= velocity_depths[-1]:
            velocities[idx] = velocity_slices[-1]
        else:
            i = np.searchsorted(velocity_depths, y) - 1
            pos_i, pos_ip1 = velocity_depths[i], velocity_depths[i + 1]
            vel_i = velocity_slices[i]
            vel_ip1 = velocity_slices[i + 1]
            beta = (
                np.log(np.abs(vel_ip1) + 1e-6) - np.log(np.abs(vel_i) + 1e-6)
            ) / (pos_ip1 - pos_i)
            alpha_val = vel_i / np.exp(beta * pos_i)
            velocities[idx] = alpha_val * np.exp(beta * y)
    return velocities