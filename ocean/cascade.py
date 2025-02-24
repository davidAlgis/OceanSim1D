import numpy as np


class WavesCascade:

    def __init__(self, N, L, wind_speed, fetch, water_depth, kmin, kmax):
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

            omega_deriv = g / (2 * omega)
            sigma = 0.07 if omega <= omega_p else 0.09
            r_exp = np.exp(-((omega - omega_p)**2) / (2 * (sigma**2) *
                                                      (omega_p**2)))
            S_omega = (alpha_js * g**2 / omega**5) * np.exp(
                -5 / 4 * (omega_p / omega)**4) * (gamma**r_exp)
            # Include continuous synthesis correction: factor delta_k².
            amplitude = np.sqrt(2 * S_omega * omega_deriv / k_abs *
                                (self.delta_k**2))
            xi = np.random.normal(0, 1) + 1j * np.random.normal(0, 1)
            self.h0[i] = xi * amplitude

        self.h0_conj = np.conjugate(self.h0)

    def update_time_dependency(self, t):
        """
        Update the time-dependent Fourier coefficients for this cascade.
        
        h̃(k,t) = h0(k) e^(i ω t) + h0_conj(-k) e^(-i ω t)
        """
        phase_plus = np.exp(1j * self.omega * t)
        phase_minus = np.exp(-1j * self.omega * t)
        h_tilde = self.h0 * phase_plus + np.conjugate(
            self.h0[self.mirror]) * phase_minus
        self.h_tilde = h_tilde
        return h_tilde

    def apply_ifft(self):
        """
        Apply the inverse FFT (with continuous correction) to recover real–space fields.
        
        Compute:
          water_height = IFFT{ h̃(k,t) * delta_k } * N,
          displacement = IFFT{ (i k/|k|) h̃(k,t) * delta_k } * N,
          derivative = IFFT{ i|k| h̃(k,t) * delta_k } * N.
        """
        h_tilde = self.h_tilde
        # Compute k_norm safely.
        abs_k = np.abs(self.k)
        k_norm = np.zeros_like(self.k)
        nonzero = abs_k >= 1e-6
        k_norm[nonzero] = self.k[nonzero] / abs_k[nonzero]
        D_tilde = 1j * k_norm * h_tilde
        derivative_hat = 1j * abs_k * h_tilde

        H = h_tilde * self.delta_k
        D_H = D_tilde * self.delta_k
        Deriv_H = derivative_hat * self.delta_k

        self.water_height = np.real(np.fft.ifft(H) * self.N)
        self.displacement = np.real(np.fft.ifft(D_H) * self.N)
        self.derivative = np.real(np.fft.ifft(Deriv_H) * self.N)

    def get_fields(self):
        """
        Return the real-space fields for this cascade.
        """
        return self.water_height, self.displacement, self.derivative
