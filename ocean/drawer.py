import sys
import pygame
import numpy as np


class OceanDrawer:

    def __init__(self,
                 ocean_instance,
                 domain_length,
                 domain_length_to_view,
                 draw_points,
                 time_scale=1.0):
        """
        Initialize the drawer with the given ocean simulation instance and display parameters.
        
        Parameters:
            ocean_instance: An instance of OceanWaves (from oceanwaves.py).
            domain_length (float): The master domain length in meters.
            domain_length_to_view (float): The simulation length (in meters) to display on screen.
            draw_points (int): Number of points on the drawing grid (can be higher than simulation grid).
            time_scale (float): Time scaling factor (unused here, fixed dt=0.016).
        """
        self.ocean = ocean_instance
        self.domain_length = domain_length
        self.domain_length_to_view = domain_length_to_view
        self.draw_points = draw_points
        self.time_scale = time_scale  # (unused, fixed dt is used)

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

    def sim_to_screen(self, x, y):
        """
        Convert simulation coordinates (x, y) to screen coordinates.
        
        x in [0, domain_length_to_view] maps to [0, screen_width]. Here we choose
        domain_length_to_view (e.g., 10 m) for display purposes.
        y in [-2, 2] maps to [top_margin, screen_height - bottom_margin],
        with higher y (more positive) toward the top of the screen.
        """
        sx = int(x / self.domain_length_to_view * self.screen_width)
        sim_y_min = -2
        sim_y_max = 2
        top_margin = 100
        bottom_margin = 100
        drawable_height = self.screen_height - top_margin - bottom_margin
        sy = top_margin + int(
            (sim_y_max - y) / (sim_y_max - sim_y_min) * drawable_height)
        return sx, sy

    def draw_grid(self):
        """
        Draw a grid on the screen to help visualize the simulation domain.
        Vertical grid lines are drawn every 1 m, and horizontal grid lines every 1 m.
        """
        grid_color = (50, 50, 50)  # dark gray
        # Draw vertical grid lines.
        x_spacing = 1  # every 1 m
        for x in np.arange(0, self.domain_length + x_spacing, x_spacing):
            sx, _ = self.sim_to_screen(x, 0)
            pygame.draw.line(self.screen, grid_color, (sx, 0),
                             (sx, self.screen_height))
        # Draw horizontal grid lines.
        y_spacing = 1  # every 1 m
        for y in np.arange(-2, 2 + y_spacing, y_spacing):
            _, sy = self.sim_to_screen(0, y)
            pygame.draw.line(self.screen, grid_color, (0, sy),
                             (self.screen_width, sy))

    def run(self):
        """
        Run the main drawing loop. The simulation is updated in real time with dt=0.016 s.
        A grid and the water surface are drawn each frame.
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

            # Display current simulation time.
            font = pygame.font.SysFont("Arial", 24)
            time_text = font.render(f"t = {t:.2f} s", True, (255, 255, 255))
            self.screen.blit(time_text, (10, 10))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
