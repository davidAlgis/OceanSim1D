import numpy as np


class OceanWaves:

    def __init__(self, N, L, wind_speed, fetch, water_depth):
        """
        Initialize the 1D ocean wave simulation parameters.
        """
        self.N = N
        self.L = L
        self.wind_speed = wind_speed
        self.fetch = fetch
        self.water_depth = water_depth  # deep water (assumed very high)

        self.g = 9.80665  # gravitational acceleration

        # Create a 1D spatial grid
        self.x = np.linspace(0, L, N)

        # Placeholders for simulation data
        self.water_height = np.zeros(N)
        self.displacement = np.zeros(N)
        self.derivative = np.zeros(N)
        self.velocity_slice = np.zeros(N)

        # Compute Fourier-space initial amplitude h0 (for each k)
        self.h0 = self.compute_h0()

    def compute_h0(self):
        """
        Compute the Fourier-space amplitude h0 (i.e., \tilde{h}_0) for each wave number
        using the Phillips spectrum with the JONSWAP frequency spectrum and a uniform directional spectrum.
        Returns:
            h0 (np.ndarray): Complex array of size N containing the initial Fourier amplitudes.
        """
        N = self.N
        L = self.L
        g = self.g

        # Wave numbers: using np.fft.fftfreq gives frequencies in cycles per unit length.
        # Multiply by 2*pi to obtain angular wave numbers.
        k = np.fft.fftfreq(N, d=L / N) * 2 * np.pi  # k can be negative!

        # Preallocate h0 array
        h0 = np.zeros(N, dtype=complex)

        # JONSWAP parameters
        U = self.wind_speed
        F = self.fetch
        alpha_js = 0.076 * ((U**2) / (F * g))**0.22
        omega_p = 22 * (g**2) / (U * F)
        gamma = 3.3

        # Loop over each mode; vectorized operations can be used but here clarity is key.
        for i in range(N):
            k_val = k[i]
            k_abs = np.abs(k_val)
            if k_abs < 1e-6:
                h0[i] = 0.0
                continue

            # Compute dispersion relation omega = sqrt(g*|k|)
            omega = np.sqrt(g * k_abs)

            # Choose sigma based on omega compared to omega_p
            sigma = 0.07 if omega <= omega_p else 0.09

            # Compute the exponent term r
            r_exp = np.exp(-((omega - omega_p)**2) / (2 * (sigma**2) *
                                                      (omega_p**2)))

            # JONSWAP frequency spectrum S(omega)
            S_omega = (alpha_js * g**2 / omega**5) * np.exp(
                -5 / 4 * (omega_p / omega)**4) * (gamma**r_exp)

            # dω/dk = g/(2ω)
            domega_dk = g / (2 * omega)

            # Directional spectrum: assuming uniform (δ=0) => D(omega)=1/(2π)
            D_omega = 1 / (2 * np.pi)

            # Combining the factors: note that the formula simplifies to:
            # h0 = xi * sqrt( (g * S(omega)) / (L * k_abs * omega) )
            amplitude = np.sqrt((g * S_omega * domega_dk *
                                 (4 * np.pi / (L * k_abs)) * D_omega))
            # Simplification: (4π/(L*k))*D_omega = (4π/(L*k))*(1/(2π)) = 2/(L*k)
            # and then multiplied by domega_dk = g/(2ω) gives:
            # amplitude = sqrt( (g * S(omega))/(L * k_abs * ω) )
            # We'll use the full expression for clarity.

            # Sample a complex Gaussian random variable (mean 0, std 1)
            xi = np.random.normal(0, 1) + 1j * np.random.normal(0, 1)

            h0[i] = xi * amplitude

        return h0

    # (Other methods: init_water_height, init_displacement, etc.)
    def init_water_height(self):
        # TODO: Compute initial water height using Fourier synthesis with self.h0.
        pass

    def init_displacement(self):
        # TODO: Compute horizontal displacement from Fourier modes.
        pass

    def init_derivative(self):
        # TODO: Compute spatial derivative of the water height.
        pass

    def init_velocity_slice(self):
        # TODO: Compute water velocity slice at a chosen depth.
        pass

    def update(self, t):
        # TODO: Update simulation state at time t.
        pass

    def run_simulation(self, total_time, dt):
        t = 0.0
        while t < total_time:
            self.update(t)
            t += dt


if __name__ == '__main__':
    # Example parameters
    N = 256  # number of grid points
    L = 100.0  # domain length in meters
    wind_speed = 20.0  # m/s
    fetch = 1000.0  # fetch in meters
    water_depth = 1e6  # deep water approximation

    ocean = OceanWaves(N, L, wind_speed, fetch, water_depth)

    # For demonstration: print first few h0 values
    print("First few h0 Fourier coefficients:")
    print(ocean.h0[:10])
