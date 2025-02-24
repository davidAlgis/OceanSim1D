import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_simulation(ocean_instance, domain_length, num_points, time_scale):
    """
    Draws and animates the simulation.
    
    This function is completely decoupled from the physics simulation.
    It receives an OceanWaves instance (from oceanwaves.py) and uses it to
    update and draw the water height on a fixed master grid.
    
    Parameters:
        ocean_instance : instance of OceanWaves (physics simulation)
        domain_length (float): The master domain length (m).
        num_points (int): Number of points on the master grid.
        time_scale (float): Time scaling factor (to control the speed of the animation).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, domain_length)
    ax.set_ylim(-2, 2)  # Adjust vertical limits as needed.
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Water Height (m)")
    title = ax.set_title("Ocean Waves at t = 0.00 s")
    line, = ax.plot([], [], lw=2, color="b")

    # Define fixed world coordinates on the master grid.
    X_world = np.linspace(0, domain_length, num_points)

    def init():
        line.set_data([], [])
        return line, title

    def update_frame(frame):
        t = frame / time_scale  # time in seconds for visualization
        ocean_instance.update(t)
        h_real = ocean_instance.get_real_water_height(X_world, N_iter=4)
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
