import numpy as np
from ocean.cascade import WavesCascade  # We still use its Fourier methods.
from ocean.init_helper import OceanInitConfig


class OceanWaves:
    # (The sea state boundaries from the original code.)
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
        Initialize the ocean simulation without cascades.
        Instead of several cascades, we create one simulation instance.
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
        y_min, y_max = self.depth_boundaries

        # Logarithmic grid for velocity interpolation.
        self.velocity_depths = self.compute_log_grid(
            y_min, y_max, self.interpolation_degree
        )

        # Master grid.
        self.x = np.linspace(0, self.master_L, self.N)

        # Instead of multiple cascades, we use a single simulation.
        # Choose kmin and kmax to include all wave modes (except very low k).
        self.kmin = 1e-6
        self.kmax = np.inf

        self.sim = WavesCascade(
            self.N,
            self.master_L,
            self.wind_speed,
            self.fetch,
            self.water_depth,
            self.kmin,
            self.kmax,
            self.interpolation_degree,
            self.velocity_depths,
        )

        # Initialize the spectrum.
        self.sim.initialize_spectrum()

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
        """
        index = min(int(round(sea_state)), len(self.SEA_STATE_BOUNDARIES) - 1)
        return self.SEA_STATE_BOUNDARIES[index]

    def compute_log_grid(self, y_min, y_max, interpolation_degree):
        """
        Generate a logarithmic depth grid for velocity computation.
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

    def update_time_dependency(self, t):
        """
        Update time-dependent Fourier coefficients.
        """
        self.sim.update_time_dependency(t)

    def apply_ifft(self):
        """
        Recover real–space fields using the inverse FFT.
        """
        self.sim.apply_ifft()

    def update(self, t):
        """
        Update simulation: time dependency and inverse FFT.
        """
        self.update_time_dependency(t)
        self.apply_ifft()

    def get_real_water_height(self, X, N_iter=4):
        """
        Compute the "real" water height at positions X.
        An iterative correction is applied.
        """
        x_guess = X.copy()
        for _ in range(N_iter):
            # Interpolate displacement from the single simulation.
            total_disp = np.interp(x_guess, self.sim.x, self.sim.displacement)
            x_guess = X - total_disp
        total_wh = np.interp(x_guess, self.sim.x, self.sim.water_height)
        return total_wh

    def get_real_water_velocity(self, X, Y):
        """
        Compute the water velocity (vx, vy) at positions (X, Y) using the
        velocity fields.
        """
        h_vals = self.get_real_water_height(X, N_iter=4)
        velocities = np.zeros((len(X), 2))
        for idx, (x, y) in enumerate(zip(X, Y)):
            # If above the water surface, velocity remains zero.
            if y > h_vals[idx]:
                continue
            # Interpolate velocity along x for each depth slice.
            velocity_slices = np.zeros((self.interpolation_degree, 2))
            for i in range(self.interpolation_degree):
                vx = np.interp(x, self.sim.x, self.sim.velocity[i, :, 0])
                vy = np.interp(x, self.sim.x, self.sim.velocity[i, :, 1])
                velocity_slices[i] = [vx, vy]

            # Depth interpolation.
            if y <= self.velocity_depths[0]:
                velocities[idx] = velocity_slices[0]
            elif y >= self.velocity_depths[-1]:
                velocities[idx] = velocity_slices[-1]
            else:
                i = np.searchsorted(self.velocity_depths, y) - 1
                pos_i, pos_ip1 = (
                    self.velocity_depths[i],
                    self.velocity_depths[i + 1],
                )
                vel_i = velocity_slices[i]
                vel_ip1 = velocity_slices[i + 1]
                # Exponential interpolation along depth.
                beta = (
                    np.log(np.abs(vel_ip1) + 1e-6)
                    - np.log(np.abs(vel_i) + 1e-6)
                ) / (pos_ip1 - pos_i)
                alpha_val = vel_i / np.exp(beta * pos_i)
                velocities[idx] = alpha_val * np.exp(beta * y)
        return velocities