import numpy as np
import pygame

from simulation.body_config import BodyConfig

class Body:
    def __init__(self):
        self.config = BodyConfig()
        self.size = self.config.body_size
        self.color = self.config.body_color
        self.antenna_length = self.config.antenna_length
        self.antenna_angle = np.radians(self.config.antenna_angle)
        self.antenna_color = self.config.antenna_color
        self.cilia_size = self.config.cilia_size
        self.cilia_color = self.config.cilia_color
        self.antennae = []
        self.cilia = []
        self.wing_color = self.config.wing_color
        self.leg_color = self.config.leg_color

    def update(self, x, y, angle):
        self.calculate_antennae(x, y, angle)
        self.calculate_cilia(x, y, angle)

    def calculate_antennae(self, x, y, angle):
        self.antennae = []
        for i in [-1, 1]:  # Left and right antennae
            antenna_x = x + np.cos(angle + i * self.antenna_angle) * self.antenna_length
            antenna_y = y + np.sin(angle + i * self.antenna_angle) * self.antenna_length
            self.antennae.append((antenna_x, antenna_y))

    def calculate_cilia(self, x, y, angle):
        self.cilia = []
        for antenna in self.antennae:
            start_x, start_y = x, y
            end_x, end_y = antenna
            for i in range(2):
                cilium_pos = (i + 1) / 3
                cilium_x = start_x + (end_x - start_x) * cilium_pos
                cilium_y = start_y + (end_y - start_y) * cilium_pos
                self.cilia.append((cilium_x, cilium_y))

    def draw(self, screen, x, y, angle):
        # Draw wings
        wing_points = [
            (-self.size, -self.size),
            (self.size, -self.size),
            (self.size * 1.5, 0),
            (self.size, self.size),
            (-self.size, self.size),
            (-self.size * 1.5, 0)
        ]
        rotated_wing_points = [self.rotate_point(p, angle) for p in wing_points]
        translated_wing_points = [(p[0] + x, p[1] + y) for p in rotated_wing_points]
        pygame.draw.polygon(screen, self.wing_color, translated_wing_points)

        # Draw body
        pygame.draw.circle(screen, self.color, (int(x), int(y)), self.size)

        # Draw legs
        for i in range(3):
            angle_offset = np.pi / 3 * i
            leg_x = np.cos(angle + angle_offset) * self.size * 1.2
            leg_y = np.sin(angle + angle_offset) * self.size * 1.2
            pygame.draw.line(screen, self.leg_color, (int(x), int(y)), (int(x + leg_x), int(y + leg_y)), 2)
            pygame.draw.line(screen, self.leg_color, (int(x), int(y)), (int(x - leg_x), int(y - leg_y)), 2)

        # Draw antennae
        for antenna in self.antennae:
            pygame.draw.line(screen, self.antenna_color, (int(x), int(y)), 
                             (int(antenna[0]), int(antenna[1])), 1)

        # Draw cilia
        for cilium in self.cilia:
            pygame.draw.circle(screen, self.cilia_color, 
                               (int(cilium[0]), int(cilium[1])), self.cilia_size)

    def rotate_point(self, point, angle):
        x, y = point
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        return (x * cos_theta - y * sin_theta,
                x * sin_theta + y * cos_theta)

    def get_antenna_positions(self):
        return self.antennae

    def get_cilia_positions(self):
        return self.cilia