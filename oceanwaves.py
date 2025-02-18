import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class OceanWaves:

    def __init__(self, N, L, wind_speed, fetch, water_depth):
        """
        Initialize the 1D ocean wave simulation parameters.
        
        Parameters:
            N (int): Number of sample points in the 1D grid.
            L (float): Domain length (m).
            wind_speed (float): Wind speed (m/s).
            fetch (float): Fetch distance (m).
            water_depth (float): Water depth (assumed very large for deep water).
        """
        self.N = N
        self.L = L
        self.wind_speed = wind_speed
        self.fetch = fetch
        self.water_depth = water_depth  # deep water assumption

        self.g = 9.80665  # gravitational acceleration (m/s^2)

        # Create a 1D spatial grid (parameter space)
        self.x = np.linspace(0, L, N)

        # Compute wave numbers using np.fft.fftfreq and convert to angular frequencies.
        self.k = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
        # Dispersion: omega = sqrt(g * |k|) (with omega[0]=0)
        self.omega = np.sqrt(self.g * np.abs(self.k))

        # Precompute indices for mirror symmetry: for each index i, mirror[i] = (-i) mod N.
        self.mirror = np.array([(-i) % N for i in range(N)])

        # Real-space fields (to be recovered via IFFT)
        self.water_height = np.zeros(N)  # h(x,t)
        self.displacement = np.zeros(N)  # horizontal displacement D(x,t)
        self.derivative = np.zeros(N)  # spatial derivative ∂h/∂x(x,t)

        # Compute the initial Fourier-space amplitudes h0.
        self.h0 = self.compute_h0()

        # Precompute Δk for the discrete synthesis.
        self.delta_k = 2 * np.pi / self.L

    def compute_h0(self):
        """
        Compute the initial Fourier amplitude h0(k) using the JONSWAP spectrum.
        
        In 2D, the theory gives:
            h0(k) = ξ(k) * sqrt((4π/(L_i * k)) S(ω) D(ω,θ) |dω/dk|),
        and for a uniform directional spectrum D(ω,θ)=1/(2π) this simplifies to
            h0(k) = ξ(k) * sqrt((g S(ω))/(L k ω)).
        
        A 1D reduction that “recovers” the energy from the continuous transform
        is obtained by including an extra factor of √(2π):
        
            h0(k) = ξ(k) * sqrt((2π g S(ω))/(L |k| ω)).
        
        The JONSWAP parameters are:
            α = 0.076 (U²/(F g))^0.22,
            ω_p = 22 (g/(U F))^(1/3)   (for dimensional consistency),
            γ = 3.3,
            σ = 0.07 if ω ≤ ω_p, 0.09 if ω > ω_p,
        and ξ(k) is a complex Gaussian variable.
        
        Returns:
            h0 (np.ndarray): Complex array of size N.
        """
        N = self.N
        L = self.L
        g = self.g
        k = self.k
        h0 = np.zeros(N, dtype=complex)

        # JONSWAP parameters.
        U = self.wind_speed
        F = self.fetch
        alpha_js = 0.076 * ((U**2) / (F * g))**0.22
        # Corrected peak frequency.
        omega_p = 22 * (g**2 / (U * F))
        gamma = 3.3

        for i in range(N):
            k_val = k[i]
            k_abs = np.abs(k_val)
            if k_abs < 1e-6:
                h0[i] = 0.0
                continue

            # Dispersion: omega = sqrt(g * |k|)
            omega = np.sqrt(g * k_abs)
            omega_derivative = np.sqrt(g * k_abs) / (2 * k_abs)
            sigma = 0.07 if omega <= omega_p else 0.09
            r_exp = np.exp(-((omega - omega_p)**2) / (2 * (sigma**2) *
                                                      (omega_p**2)))
            S_omega = (alpha_js * g**2 / omega**5) * np.exp(
                -5 / 4 * (omega_p / omega)**4) * (gamma**r_exp)

            # Amplitude (with the extra √(2π) factor).
            delta_k = 2 * np.pi / L
            amplitude = np.sqrt(2 * S_omega * abs(omega_derivative) / k_abs *
                                delta_k**2)
            # Sample a complex Gaussian variable (mean 0, std dev 1 for each component).
            xi = np.random.normal(0, 1) + 1j * np.random.normal(0, 1)
            h0[i] = xi * amplitude

        return h0

    def update(self, t):
        """
        Update the simulation state at time t.

        Computes the time-dependent Fourier coefficients:
            h̃(k,t) = h0(k) e^(i ω t) + h0*(-k) e^(-i ω t),
        then synthesizes the spatial fields via the inverse FFT with continuous 
        synthesis correction.
        
        The horizontal displacement is computed as:
            D(x,t) = IFFT{ (i * k/|k|) h̃(k,t) },
        with the convention that for k=0 we set the factor to 0.
        
        The derivative of h is computed as:
            ∂h/∂x (x,t) = IFFT{ i |k| h̃(k,t) }.
        """
        phase_plus = np.exp(1j * self.omega * t)
        phase_minus = np.exp(-1j * self.omega * t)

        # Time-dependent Fourier coefficients.
        h_tilde = self.h0 * phase_plus + np.conjugate(
            self.h0[self.mirror]) * phase_minus

        # For displacement: compute k_norm = k/|k|, with safe handling for k = 0.
        k_norm = np.where(np.abs(self.k) < 1e-6, 0, self.k / np.abs(self.k))
        D_tilde = 1j * k_norm * h_tilde

        # For the derivative of h, use: derivative_hat = i * |k| * h_tilde.
        derivative_hat = 1j * np.abs(self.k) * h_tilde

        # Multiply by Δk to approximate the continuous integral.
        H = h_tilde * self.delta_k
        D_H = D_tilde * self.delta_k
        Deriv_H = derivative_hat * self.delta_k

        # Use the inverse FFT. (NumPy’s ifft includes a 1/N factor, so we multiply by N.)
        self.water_height = np.real(np.fft.ifft(H) * self.N)
        self.displacement = np.real(np.fft.ifft(D_H) * self.N)
        self.derivative = np.real(np.fft.ifft(Deriv_H) * self.N)

    def get_real_water_height(self, X, N_iter=4):
        """
        Retrieve the "real" water height at fixed world coordinates X.
        
        The simulated water surface is given parametrically by
            ( x + D(x,t), h(x,t) ).
        To obtain the water height at a world coordinate X, we iteratively solve:
            x* = X - D(x*,t),
        then set h_real(X,t) = h(x*,t).
        
        Parameters:
            X (np.ndarray): World positions.
            N_iter (int): Number of iterations.
            
        Returns:
            h_real (np.ndarray): Water height at positions X.
        """
        x_guess = X.copy()
        for _ in range(N_iter):
            D_guess = np.interp(x_guess, self.x, self.displacement)
            x_guess = X - D_guess
        h_real = np.interp(x_guess, self.x, self.water_height)
        return h_real


def animate_wave():
    # Simulation parameters.
    N = 256  # Number of grid points.
    L = 10.0  # Domain length in meters.
    wind_speed = 5.0  # m/s.
    fetch = 1000.0  # Fetch in meters.
    water_depth = 1e6  # Deep water.

    # Create an instance of the simulation.
    ocean = OceanWaves(N, L, wind_speed, fetch, water_depth)

    # Set up the Matplotlib figure.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 5)
    ax.set_ylim(-2,
                2)  # Adjust vertical limits to see realistic wave amplitudes.
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Water Height (m)")
    title = ax.set_title("Ocean Waves at t = 0.00 s")
    line, = ax.plot([], [], lw=2, color="b")

    # Fixed world coordinates where we evaluate the "real" water height.
    X_world = np.linspace(0, L, N)

    def init():
        line.set_data([], [])
        return line, title

    def update_frame(frame):
        t = frame / 10.0  # Time (in seconds) for visualization.
        ocean.update(t)
        h_real = ocean.get_real_water_height(X_world, N_iter=4)
        line.set_data(X_world, h_real)
        title.set_text(f"Ocean Waves at t = {t:.2f} s")
        return line, title

    ani = animation.FuncAnimation(fig,
                                  update_frame,
                                  frames=200,
                                  init_func=init,
                                  blit=True,
                                  interval=50)
    plt.show()


if __name__ == '__main__':
    animate_wave()
