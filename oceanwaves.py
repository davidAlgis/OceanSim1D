import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class OceanWaves:

    def __init__(self, N, L, wind_speed, fetch, water_depth):
        """
        Initialize the 1D ocean wave simulation parameters.
        
        Parameters:
            N (int): Number of sample points in the 1D grid.
            L (float): Domain length.
            wind_speed (float): Wind speed in m/s.
            fetch (float): Fetch in meters.
            water_depth (float): Water depth (assumed very large for deep water).
        """
        self.N = N
        self.L = L
        self.wind_speed = wind_speed
        self.fetch = fetch
        self.water_depth = water_depth  # Deep water assumption

        self.g = 9.80665  # gravitational acceleration

        # Create a 1D spatial grid (parameter space)
        self.x = np.linspace(0, L, N)

        # Compute wave numbers (using np.fft.fftfreq) and angular frequencies.
        self.k = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
        self.omega = np.sqrt(self.g * np.abs(self.k))

        # For accessing negative wave numbers via index (mirror symmetry)
        self.mirror = np.array([(-i) % N for i in range(N)])

        # Placeholders for real-space fields
        self.water_height = np.zeros(N)  # h(x, t) from IFFT
        self.displacement = np.zeros(N)  # horizontal displacement D(x, t)
        self.derivative = np.zeros(N)  # spatial derivative of h

        # Precompute the Fourier-space initial amplitudes h0 for each mode.
        self.h0 = self.compute_h0()

    def compute_h0(self):
        """
        Compute the initial Fourier-space amplitude h0(k) using the JONSWAP frequency spectrum
        (with a uniform directional spectrum, i.e., δ=0) as described in the article.
        
        In 1D, we define:
            h0(k) = xi * sqrt((g * S(omega)) / (L * |k| * omega))
        where:
            - xi is a complex Gaussian random variable,
            - omega = sqrt(g * |k|),
            - S(omega) = (alpha * g^2 / omega^5) exp[-(5/4)(omega_p/omega)^4] gamma^r,
              with r = exp[-((omega-omega_p)^2)/(2*sigma^2*omega_p^2)]
            - alpha = 0.076 * ((U^2)/(F*g))^0.22,
            - omega_p = 22*g^2/(U*F),
            - gamma = 3.3,
            - sigma = 0.07 if omega<=omega_p else 0.09.
        """
        N = self.N
        L = self.L
        g = self.g
        k = self.k
        h0 = np.zeros(N, dtype=complex)

        # JONSWAP parameters
        U = self.wind_speed
        F = self.fetch
        alpha_js = 0.076 * ((U**2) / (F * g))**0.22
        omega_p = 22 * (g**2) / (U * F)
        gamma = 3.3

        for i in range(N):
            k_val = k[i]
            k_abs = np.abs(k_val)
            if k_abs < 1e-6:
                h0[i] = 0.0
                continue

            # Compute dispersion: omega = sqrt(g*|k|)
            omega = np.sqrt(g * k_abs)
            sigma = 0.07 if omega <= omega_p else 0.09
            r_exp = np.exp(-((omega - omega_p)**2) / (2 * (sigma**2) *
                                                      (omega_p**2)))
            S_omega = (alpha_js * g**2 / omega**5) * np.exp(
                -5 / 4 * (omega_p / omega)**4) * (gamma**r_exp)

            # Note: dω/dk = g/(2ω) is absorbed in the derivation.
            amplitude = np.sqrt((g * S_omega) / (L * k_abs * omega))

            # Sample a complex Gaussian variable (mean 0, std dev 1)
            xi = np.random.normal(0, 1) + 1j * np.random.normal(0, 1)
            h0[i] = xi * amplitude

        return h0

    def update(self, t):
        """
        Update the simulation state at time t.
        
        Computes the time-dependent Fourier coefficients:
            h̃(k,t) = h0(k)e^(iωt) + h0*(-k)e^(-iωt)
        and then uses IFFT to compute:
            - water_height h(x,t)
            - horizontal displacement D(x,t) [computed as i*sign(k)*h̃(k,t)]
            - derivative (∂h/∂x)(x,t)
        """
        N = self.N
        phase_plus = np.exp(1j * self.omega * t)
        phase_minus = np.exp(-1j * self.omega * t)

        # h̃(k,t) (using mirror symmetry to obtain h0(-k))
        h_tilde = self.h0 * phase_plus + np.conjugate(
            self.h0[self.mirror]) * phase_minus

        # Horizontal displacement in Fourier space: D̃(k,t) = i * sign(k) * h̃(k,t)
        sign_k = np.sign(self.k)
        D_tilde = 1j * sign_k * h_tilde

        # Fourier-space derivative: i * k * h̃(k,t)
        derivative_hat = 1j * self.k * h_tilde

        # Inverse FFT to return to real space.
        self.water_height = np.real(np.fft.ifft(h_tilde))
        self.displacement = np.real(np.fft.ifft(D_tilde))
        self.derivative = np.real(np.fft.ifft(derivative_hat))

    def get_real_water_height(self, X, N_iter=4):
        """
        Retrieve the 'real' water height at fixed world positions X.
        
        The simulated water surface is given parametrically as:
            ( x + D(x,t), h(x,t) )
        so to find the water height at a given world position X, we iteratively solve:
            x* = X - D(x*, t)
        and then h_real(X,t) = h(x*, t).
        
        Parameters:
            X (np.ndarray): Array of fixed world coordinates.
            N_iter (int): Number of iterations for convergence.
            
        Returns:
            h_real (np.ndarray): Water height evaluated at positions X.
        """
        # Initial guess: use X as the parameter value.
        x_guess = X.copy()
        for _ in range(N_iter):
            # Interpolate the horizontal displacement at x_guess.
            D_guess = np.interp(x_guess, self.x, self.displacement)
            # Update the guess: x* = X - D(x*, t)
            x_guess = X - D_guess
        # Finally, interpolate the water height using the converged parameter values.
        h_real = np.interp(x_guess, self.x, self.water_height)
        return h_real


def animate_wave():
    # Simulation parameters
    N = 4096
    L = 1000.0
    wind_speed = 20.0
    fetch = 10000.0
    water_depth = 1e6  # deep water

    # Create an OceanWaves instance.
    ocean = OceanWaves(N, L, wind_speed, fetch, water_depth)

    # Set up the Matplotlib figure.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, L)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("x")
    ax.set_ylabel("Water Height")
    title = ax.set_title("Ocean Waves at t = 0.00 s")
    line, = ax.plot([], [], lw=2, color="b")

    # Define a set of fixed world coordinates (for which we want the real water height).
    X_world = np.linspace(0, L, N)

    def init():
        line.set_data([], [])
        return line, title

    def update_frame(frame):
        t = frame / 10.0  # Scale time for visualization
        ocean.update(t)
        # Compute the real water height at world positions X_world.
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
