import sys
import pygame
import numpy as np


class OceanDrawer:

    def __init__(self, ocean_instance, domain_length, domain_length_to_view,
                 draw_points, time_scale, interpolation_degree):
        """
        Initialize the drawer with the given ocean simulation instance and display parameters.
        
        Parameters:
            ocean_instance: An instance of OceanWaves (from oceanwaves.py).
            domain_length (float): The master domain length in meters.
            domain_length_to_view (float): The simulation length (in meters) to display on screen.
            draw_points (int): Number of points on the drawing grid (can be higher than simulation grid).
            time_scale (float): Time scaling factor (unused here, fixed dt=0.016).
            interpolation_degree (int): Number of depth levels for velocity computation.
        """
        self.ocean = ocean_instance
        self.domain_length = domain_length
        self.domain_length_to_view = domain_length_to_view
        self.draw_points = draw_points
        self.time_scale = time_scale  # (unused, fixed dt is used)
        self.interpolation_degree = interpolation_degree

        # Pygame display setup.
        self.screen_width = 800
        self.screen_height = 600
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height))
        pygame.display.set_caption("1D Tessendorf Ocean Waves")
        self.clock = pygame.time.Clock()

        # Create a fixed master grid for drawing (using higher resolution).
        self.X_world = np.linspace(0, domain_length, draw_points)

        # Define velocity grid: 1m spacing horizontally, 0.25m vertically
        self.velocity_grid_x = np.arange(0, self.domain_length_to_view, 1.0)
        self.velocity_grid_y = np.arange(2, -4, -1)

    def sim_to_screen(self, x, y):
        """
        Convert simulation coordinates (x, y) to screen coordinates.
        
        x in [0, domain_length_to_view] maps to [0, screen_width].
        y in [-3, 1] maps to [top_margin, screen_height - bottom_margin].
        """
        sx = int(x / self.domain_length_to_view * self.screen_width)
        sim_y_min = -3
        sim_y_max = 1
        top_margin = 100
        bottom_margin = 100
        drawable_height = self.screen_height - top_margin - bottom_margin
        sy = top_margin + int(
            (sim_y_max - y) / (sim_y_max - sim_y_min) * drawable_height)
        return sx, sy

    def draw_grid(self):
        """
        Draw a grid on the screen to help visualize the simulation domain.
        Vertical grid lines are drawn every 1m, horizontal every 0.25m.
        """
        grid_color = (50, 50, 50)  # dark gray
        # Draw vertical grid lines.
        for x in self.velocity_grid_x:
            sx, _ = self.sim_to_screen(x, 0)
            pygame.draw.line(self.screen, grid_color, (sx, 0),
                             (sx, self.screen_height))
        # Draw horizontal grid lines.
        for y in self.velocity_grid_y:
            _, sy = self.sim_to_screen(0, y)
            pygame.draw.line(self.screen, grid_color, (0, sy),
                             (self.screen_width, sy))

    def draw_velocity_vectors(self, velocities):
        """
        Draw velocity vectors at pre-defined grid points.

        Parameters:
            velocities (np.ndarray): Array of shape (Nx, Ny, 2) storing (vx, vy).
        """
        arrow_color = (255, 0, 0)  # Red for velocity arrows
        scale_factor = 5  # Scale velocity for visibility

        for i, x in enumerate(self.velocity_grid_x):
            for j, y in enumerate(self.velocity_grid_y):
                sx, sy = self.sim_to_screen(x, y)
                vx, vy = velocities[i, j] * scale_factor

                # Compute arrow endpoint
                ex = sx + vx
                ey = sy - vy  # Invert Y-axis for pygame

                # Draw velocity vector
                pygame.draw.line(self.screen, arrow_color, (sx, sy), (ex, ey),
                                 2)

    def run(self):
        """
        Run the main drawing loop. The simulation is updated in real time with dt=0.016 s.
        A grid, water surface, and velocity vectors are drawn each frame.
        """
        t = 0.0
        dt = 0.016  # fixed time increment (approx 60 FPS)
        running = True
        while running:
            # Process events.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update simulation.
            t += dt
            self.ocean.update(t)
            h_real = self.ocean.get_real_water_height(self.X_world, N_iter=4)

            # Compute velocity field at grid points
            X_grid, Y_grid = np.meshgrid(self.velocity_grid_x,
                                         self.velocity_grid_y,
                                         indexing='ij')
            velocities = self.ocean.get_real_water_velocity(
                X_grid.ravel(), Y_grid.ravel())
            velocities = velocities.reshape(
                (len(self.velocity_grid_x), len(self.velocity_grid_y), 2))
            # print(velocities)

            # Clear screen and draw grid.
            self.screen.fill((30, 30, 30))
            self.draw_grid()

            # Map simulation points to screen coordinates.
            points = [
                self.sim_to_screen(x_val, h)
                for x_val, h in zip(self.X_world, h_real)
            ]
            if len(points) >= 2:
                pygame.draw.lines(self.screen, (0, 255, 255), False, points, 2)

            # Draw velocity vectors
            self.draw_velocity_vectors(velocities)

            # Display current simulation time.
            font = pygame.font.SysFont("Arial", 24)
            time_text = font.render(f"t = {t:.2f} s", True, (255, 255, 255))
            self.screen.blit(time_text, (10, 10))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == '__main__':
    from ocean.oceanwaves import OceanWaves
    ocean_sim = OceanWaves(256, 100.0, 5.0, 1000.0, 1e6, 8)
    drawer = OceanDrawer(ocean_sim, 100.0, 10.0, 512, 1.0, 8)
    drawer.run()
