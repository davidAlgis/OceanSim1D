import numpy as np


class WavesCascade:

    def __init__(
        self,
        N,
        L,
        wind_speed,
        fetch,
        water_depth,
        kmin,
        kmax,
        interpolation_degree,
        log_grid,
    ):
        """
        Initialize a single cascade.

        Parameters:
            N (int): Number of grid points for this cascade.
            L (float): Cascade domain length (L_i).
            wind_speed (float): Wind speed (m/s).
            fetch (float): Fetch (m).
            water_depth (float): Water depth (assumed deep).
            kmin (float): Minimum absolute wave number for this cascade.
            kmax (float): Maximum absolute wave number for this cascade (use
            np.inf for no upper cutoff). interpolation_degree (int): Number of
            depth levels for velocity computation. log_grid (np.ndarray):
        Logarithmic depth grid.
        """
        self.N = N
        self.L = L
        self.wind_speed = wind_speed
        self.fetch = fetch
        self.water_depth = water_depth
        self.interpolation_degree = interpolation_degree
        self.log_grid = log_grid
        self.g = 9.80665

        # Spatial grid for this cascade.
        self.x = np.linspace(0, L, N)

        # Fourier space.
        self.k = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
        self.omega = np.sqrt(self.g * np.abs(self.k))
        self.mirror = np.array([(-i) % N for i in range(N)])
        self.delta_k = 2 * np.pi / L

        # k–band cutoffs.
        self.kmin = kmin
        self.kmax = kmax

        # Spectrum arrays.
        self.h0 = np.zeros(N, dtype=complex)
        self.h0_conj = np.zeros(N, dtype=complex)

        # Real–space fields.
        self.water_height = np.zeros(N)
        self.h_tilde = np.zeros(N, dtype=complex)
        self.displacement = np.zeros(N)
        self.displacement_tilde = np.zeros(N, dtype=complex)
        self.derivative = np.zeros(N)
        self.derivative_tilde = np.zeros(N, dtype=complex)

        # Velocity fields (2D array for different depth slices).
        self.velocity = np.zeros((interpolation_degree, N, 2))
        self.velocity_tilde = np.zeros(
            (interpolation_degree, N, 2), dtype=complex
        )

    def initialize_spectrum(self):
        """
        Compute the initial Fourier amplitude h0(k) using the JONSWAP spectrum,
        but only for modes with |k| in [kmin, kmax). Also compute the
        conjugate.
        """
        N = self.N
        L = self.L
        g = self.g
        k_arr = self.k
        self.h0 = np.zeros(N, dtype=complex)

        # JONSWAP parameters.
        U = self.wind_speed
        F = self.fetch
        alpha_js = 0.076 * ((U**2) / (F * g)) ** 0.22
        omega_p = 22 * (g**2 / (U * F))
        gamma = 3.3

        for i in range(N):
            k_val = k_arr[i]
            k_abs = np.abs(k_val)
            if k_abs < 1e-6 or k_abs < self.kmin or k_abs >= self.kmax:
                self.h0[i] = 0.0
                continue

            omega = np.sqrt(g * k_abs)
            # dω/dk = g/(2ω)
            omega_deriv = g / (2 * omega)
            sigma = 0.07 if omega <= omega_p else 0.09
            r_exp = np.exp(
                -((omega - omega_p) ** 2) / (2 * (sigma**2) * (omega_p**2))
            )
            S_omega = (
                (alpha_js * g**2 / omega**5)
                * np.exp(-5 / 4 * (omega_p / omega) ** 4)
                * (gamma**r_exp)
            )

            # Continuous synthesis correction: include factor delta_k².
            amplitude = np.sqrt(
                2 * S_omega * omega_deriv / k_abs * (self.delta_k**2)
            )
            # Add a seed to make it deterministic
            np.random.seed(42)
            xi = np.random.normal(0, 1) + 1j * np.random.normal(0, 1)
            self.h0[i] = xi * amplitude

        self.h0_conj = np.conjugate(self.h0)

    def update_time_dependency(self, t):
        """
        Update the time-dependent Fourier coefficients for this cascade,
        computing displacement, derivative, and velocity in Fourier space.
        """
        phase = self.omega * t
        exponent = np.exp(1j * phase)
        exponentConj = np.exp(-1j * phase)

        h = (
            self.h0 * exponent
            + np.conjugate(self.h0[self.mirror]) * exponentConj
        )
        h_v = (
            self.h0 * exponent
            - np.conjugate(self.h0[self.mirror]) * exponentConj
        )
        ih = -np.imag(h) + 1j * np.real(h)

        abs_k = np.abs(self.k)
        k_norm = np.divide(
            self.k, abs_k, out=np.zeros_like(self.k), where=(abs_k >= 1e-6)
        )

        # Compute Displacement, Derivative, and Velocity in Fourier space
        self.h_tilde = h * self.delta_k
        self.displacement_tilde = ih * k_norm * self.delta_k
        self.derivative_tilde = 1j * abs_k * h * self.delta_k

        # Velocity computation at different depth slices
        safe_omega = np.divide(
            self.g,
            self.omega,
            out=np.zeros_like(self.omega),
            where=(self.omega >= 1e-6),
        )

        for i, depth in enumerate(self.log_grid):
            if depth > 0:
                attenuation = 1 + abs_k * depth
            else:
                attenuation = np.exp(abs_k * depth)

            vx_tilde = -self.k * safe_omega * h_v * attenuation * self.delta_k
            vy_tilde = 1j * self.omega * h_v * attenuation * self.delta_k

            # self.velocity[i, :, 0] = np.real(vx_tilde)
            self.velocity_tilde[i, :, 0] = vx_tilde
            self.velocity_tilde[i, :, 1] = vy_tilde

    def apply_ifft(self):
        """
        Apply the inverse FFT (with continuous synthesis correction) to recover
        real–space fields.
        """
        self.water_height = np.real(np.fft.ifft(self.h_tilde) * self.N)
        self.displacement = np.real(
            np.fft.ifft(self.displacement_tilde) * self.N
        )
        self.derivative = np.real(np.fft.ifft(self.derivative_tilde) * self.N)

        # Apply IFFT to velocity at each depth slice
        for i in range(self.interpolation_degree):
            vx_ifft = np.real(
                np.fft.ifft(self.velocity_tilde[i, :, 0]) * self.N
            )
            vy_ifft = np.real(
                np.fft.ifft(self.velocity_tilde[i, :, 1]) * self.N
            )
            self.velocity[i, :, 0] = vx_ifft
            self.velocity[i, :, 1] = vy_ifft

    def get_fields(self):
        """
        Return the real-space fields for this cascade.
        """
        return self.water_height, self.displacement, self.derivative

    def get_velocity_at_depth(self, depth_index):
        """
        Retrieve the velocity field at a specific depth slice.
        """
        return self.velocity[depth_index, :, :]