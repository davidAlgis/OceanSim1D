import sys
import pygame
import numpy as np


def animate_simulation(ocean_instance, domain_length, num_points, time_scale):
    """
    Draws and animates the simulation using pygame.
    
    This function is completely decoupled from the physics simulation.
    It receives an OceanWaves instance (from oceanwaves.py) and uses it to
    update and draw the water height on a fixed master grid.
    
    Parameters:
        ocean_instance : instance of OceanWaves (physics simulation)
        domain_length (float): The master domain length (m).
        num_points (int): Number of points on the master grid.
        time_scale (float): Time scaling factor (to control the speed of the animation).
    """
    # Initialize Pygame.
    pygame.init()
    screen_width = 800
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("1D Tessendorf Ocean Waves")
    clock = pygame.time.Clock()

    # Define a fixed master grid in world coordinates.
    X_world = np.linspace(0, domain_length, num_points)

    # Define a helper function to convert simulation coordinates to screen coordinates.
    def sim_to_screen(x, y):
        # Map x from [0, domain_length] to [0, screen_width].
        sx = int(x / domain_length * screen_width)
        # Assume simulation water height is in roughly [-2, 2] (m).
        # We'll map y = 2 (highest) to a top margin (say, 100) and y = -2 to a bottom margin (say, 500).
        sim_y_min = -2
        sim_y_max = 2
        top_margin = 100
        bottom_margin = 100
        drawable_height = screen_height - top_margin - bottom_margin
        # In pygame, y increases downward, so higher simulation y should yield a smaller screen y.
        sy = top_margin + int(
            (sim_y_max - y) / (sim_y_max - sim_y_min) * drawable_height)
        return sx, sy

    # Simulation time variable.
    t = 0.0
    dt = 0.05  # Simulation time increment per frame (adjust as needed).

    running = True
    while running:
        # Process events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update simulation.
        t += dt
        ocean_instance.update(t)
        h_real = ocean_instance.get_real_water_height(X_world, N_iter=4)

        # Clear the screen.
        screen.fill((30, 30, 30))  # dark gray background

        # Convert simulation points to screen coordinates.
        points = []
        for x_val, h in zip(X_world, h_real):
            points.append(sim_to_screen(x_val, h))

        # Draw the water surface as a connected line.
        if len(points) >= 2:
            pygame.draw.lines(screen, (0, 255, 255), False, points,
                              2)  # cyan line

        # Optionally, display the current simulation time.
        font = pygame.font.SysFont("Arial", 24)
        time_text = font.render(f"t = {t:.2f} s", True, (255, 255, 255))
        screen.blit(time_text, (10, 10))

        # Update the display.
        pygame.display.flip()
        clock.tick(60)  # Limit to 60 FPS

    pygame.quit()


if __name__ == '__main__':
    # For testing purposes you can import your physics simulation (OceanWaves) here.
    # Otherwise, this function is intended to be called from main.py.
    from oceanwaves import OceanWaves
    # Use default parameters.
    ocean_sim = OceanWaves(256, 100.0, 5.0, 1000.0, 1e6)
    animate_simulation(ocean_sim, 100.0, 256, 10.0)
