import numpy as np


class WavesCascade:

    def __init__(self, N, L, wind_speed, fetch, water_depth, kmin, kmax,
                 interpolation_degree):
        """
        Initialize a single cascade.
        
        Parameters:
            N (int): Number of grid points for this cascade.
            L (float): Cascade domain length (L_i).
            wind_speed (float): Wind speed (m/s).
            fetch (float): Fetch (m).
            water_depth (float): Water depth (assumed deep).
            kmin (float): Minimum absolute wave number for this cascade.
            kmax (float): Maximum absolute wave number for this cascade (use np.inf for no upper cutoff).
        """
        self.N = N
        self.L = L
        self.wind_speed = wind_speed
        self.fetch = fetch
        self.water_depth = water_depth
        self.interpolation_degree = interpolation_degree
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
        self.displacement = np.zeros(N)
        self.derivative = np.zeros(N)

    def initialize_spectrum(self):
        """
        Compute the initial Fourier amplitude h0(k) using the JONSWAP spectrum,
        but only for modes with |k| in [kmin, kmax). Also compute the conjugate.
        """
        N = self.N
        L = self.L
        g = self.g
        k_arr = self.k
        self.h0 = np.zeros(N, dtype=complex)

        # JONSWAP parameters.
        U = self.wind_speed
        F = self.fetch
        alpha_js = 0.076 * ((U**2) / (F * g))**0.22
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
            r_exp = np.exp(-((omega - omega_p)**2) / (2 * (sigma**2) *
                                                      (omega_p**2)))
            S_omega = (alpha_js * g**2 / omega**5) * np.exp(
                -5 / 4 * (omega_p / omega)**4) * (gamma**r_exp)
            # Continuous synthesis correction: include factor delta_k².
            amplitude = np.sqrt(2 * S_omega * omega_deriv / k_abs *
                                (self.delta_k**2))
            xi = np.random.normal(0, 1) + 1j * np.random.normal(0, 1)
            self.h0[i] = xi * amplitude

        self.h0_conj = np.conjugate(self.h0)

    def update_time_dependency(self, t):
        """
        Update the time-dependent Fourier coefficients for this cascade,
        computing displacement and derivative in Fourier space, mimicking the CUDA kernel.
        """
        phase = self.omega * t
        exponent = np.cos(phase) + 1j * np.sin(phase)
        exponentConj = np.cos(phase) - 1j * np.sin(phase)
        h = self.h0 * exponent + np.conjugate(
            self.h0[self.mirror]) * exponentConj

        # Compute ih = -Im(h) + i * Re(h)
        ih = -np.imag(h) + 1j * np.real(h)

        abs_k = np.abs(self.k)
        # Use np.divide to avoid warnings
        k_norm = np.divide(self.k,
                           abs_k,
                           out=np.zeros_like(self.k),
                           where=(abs_k >= 1e-6))

        self.h_tilde = h * self.delta_k
        self.displacement_tilde = ih * k_norm * self.delta_k
        self.derivative_tilde = 1j * abs_k * h * self.delta_k

    def apply_ifft(self):
        """
        Apply the inverse FFT (with continuous synthesis correction) to recover real–space fields.
        
        Computes:
          water_height = IFFT{ h_tilde * delta_k } * N,
          displacement = IFFT{ (computed displacement_tilde) * delta_k } * N,
          derivative = IFFT{ (computed derivative_tilde) * delta_k } * N.
        
        Then post-process the result by multiplying by (-1)^i (for each spatial index i)
        to account for the transform range being [-N/2, N/2].
        Finally, apply a results merger to the displacement.
        """
        wh = np.real(np.fft.ifft(self.h_tilde) * self.N)
        disp = np.real(np.fft.ifft(self.displacement_tilde) * self.N)
        deriv = np.real(np.fft.ifft(self.derivative_tilde) * self.N)

        # Multiply each output by the corresponding multiplier.
        self.water_height = np.real(wh)
        self.displacement = np.real(disp)
        self.derivative = np.real(deriv)

    def get_fields(self):
        """
        Return the real-space fields for this cascade.
        """
        return self.water_height, self.displacement, self.derivative
