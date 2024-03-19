# This will handle the body, including receptor structures
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

class Body:
    def __init__(self):
        self.size = 12

        self.antenna_angle = 55 * (np.pi / 180)
        self.antenna_length = 12 * 5
        self.antenna_width = 5
        self.antenna_color = (0, 200, 0)

        self.cilia_size = self.size * 0.1
        self.cilia_width = 0.25
        self.cilia_color = (0, 200, 0)

        self.antennae = []
        self.cilia = []

        self.tail_wiggle_angle = 15 * np.pi / 180  # Maximum wiggle angle deviation
        self.tail_wiggle_speed = 2  # Speed of the wiggle
        self.wiggle_phase = 0  # Current phase of the wiggle
        self.tail_angle = 0

    def draw(self, screen, x, y, angle):
        angle = angle

        # FIRST CIRCLE  
        pygame.draw.circle(screen, (0, 255, 0), (int(x), int(y)), self.size)
        
        # SECOND CIRCLE
        recessed_x1 = x - np.cos(angle) * self.size * 0.75
        recessed_y1 = y - np.sin(angle) * self.size * 0.75
        pygame.draw.circle(screen, (0, 255, 0), (int(recessed_x1), int(recessed_y1)), self.size / 1.5)

        # THIRD CIRCLE
        recessed_x2 = recessed_x1 - np.cos(angle) * self.size * 0.75
        recessed_y2 = recessed_y1 - np.sin(angle) * self.size * 0.75
        pygame.draw.circle(screen, (0, 255, 0), (int(recessed_x2), int(recessed_y2)), self.size / 2)

        # FIRST ANTENNAE
        forward_shift = 0
        left_antenna_x = x + np.cos(angle + self.antenna_angle) * self.antenna_length + np.cos(angle) * forward_shift
        self.L1_antenna_x = left_antenna_x
        left_antenna_y = y + np.sin(angle + self.antenna_angle) * self.antenna_length + np.sin(angle) * forward_shift
        self.L1_antenna_y = left_antenna_y
        right_antenna_x = x + np.cos(angle - self.antenna_angle) * self.antenna_length + np.cos(angle) * forward_shift
        self.R1_antenna_x = right_antenna_x
        right_antenna_y = y + np.sin(angle - self.antenna_angle) * self.antenna_length + np.sin(angle) * forward_shift
        self.R1_antenna_y = right_antenna_y
        pygame.draw.line(screen, (0, 255, 0), (x, y), (left_antenna_x, left_antenna_y), 1)
        pygame.draw.line(screen, (0, 255, 0), (x, y), (right_antenna_x, right_antenna_y), 1)

        # SECOND ANTENNAE
        backward_shift = 0
        left_antenna2_x = recessed_x1 + np.cos(angle + self.antenna_angle) * self.antenna_length - np.cos(angle) * backward_shift
        self.L2_antenna_x = left_antenna2_x
        left_antenna2_y = recessed_y1 + np.sin(angle + self.antenna_angle) * self.antenna_length - np.sin(angle) * backward_shift
        self.L2_antenna_y = left_antenna2_y
        right_antenna2_x = recessed_x1 + np.cos(angle - self.antenna_angle) * self.antenna_length - np.cos(angle) * backward_shift
        self.R2_antenna_x = right_antenna2_x
        right_antenna2_y = recessed_y1 + np.sin(angle - self.antenna_angle) * self.antenna_length - np.sin(angle) * backward_shift
        self.R2_antenna_y = right_antenna2_y
        pygame.draw.line(screen, (0, 255, 0), (recessed_x1, recessed_y1), (left_antenna2_x, left_antenna2_y), 1)
        pygame.draw.line(screen, (0, 255, 0), (recessed_x1, recessed_y1), (right_antenna2_x, right_antenna2_y), 1)

        # TAIL
        wiggle_effect = np.sin(self.wiggle_phase) * self.tail_wiggle_angle
        self.tail_angle = angle + wiggle_effect
        tail_length = self.size * 1.5
        tail_width = self.size * 0.5  # Adjust the width of the tail triangle

        # Calculate the three vertices of the tail triangle
        tail_tip_x = recessed_x2 - np.cos(self.tail_angle) * tail_length
        tail_tip_y = recessed_y2 - np.sin(self.tail_angle) * tail_length
        tail_left_x = recessed_x2 - np.cos(self.tail_angle + np.pi/2) * tail_width/2
        tail_left_y = recessed_y2 - np.sin(self.tail_angle + np.pi/2) * tail_width/2
        tail_right_x = recessed_x2 - np.cos(self.tail_angle - np.pi/2) * tail_width/2
        tail_right_y = recessed_y2 - np.sin(self.tail_angle - np.pi/2) * tail_width/2

        # Draw the tail triangle
        pygame.draw.polygon(screen, (0, 255, 0), [(tail_tip_x, tail_tip_y), (tail_left_x, tail_left_y), (tail_right_x, tail_right_y)])

        # CILIA
        for cilium in self.cilia:
            pygame.draw.circle(screen, self.cilia_color, (int(cilium[0]), int(cilium[1])), self.cilia_size)

    def wiggle(self, distance):
        self.wiggle_phase += self.tail_wiggle_speed * distance

    def get_antennae(self, x, y, angle):
        # Calculate and update the positions of all antennae based on the current entity position and angle
        self.L1_antenna_x = x + np.cos(angle + self.antenna_angle) * self.antenna_length
        self.L1_antenna_y = y + np.sin(angle + self.antenna_angle) * self.antenna_length
        self.R1_antenna_x = x + np.cos(angle - self.antenna_angle) * self.antenna_length
        self.R1_antenna_y = y + np.sin(angle - self.antenna_angle) * self.antenna_length

        recessed_x1 = x - np.cos(angle) * self.size * 0.75
        recessed_y1 = y - np.sin(angle) * self.size * 0.75

        self.L2_antenna_x = recessed_x1 + np.cos(angle + self.antenna_angle) * self.antenna_length
        self.L2_antenna_y = recessed_y1 + np.sin(angle + self.antenna_angle) * self.antenna_length
        self.R2_antenna_x = recessed_x1 + np.cos(angle - self.antenna_angle) * self.antenna_length
        self.R2_antenna_y = recessed_y1 + np.sin(angle - self.antenna_angle) * self.antenna_length

        # Update antennae list
        self.antennae = [[self.L1_antenna_x, self.L1_antenna_y], [self.R1_antenna_x, self.R1_antenna_y],
                    [self.L2_antenna_x, self.L2_antenna_y], [self.R2_antenna_x, self.R2_antenna_y]]

        return self.antennae

    def get_cilia(self, x, y):
        # Calculate and update the positions of all cilia based on the c    # Cilia are receptors placed at the tip and halfway point of each antenna

        self.cilia = []

        for antenna in self.antennae:
            start_x, start_y = x, y
            end_x, end_y = antenna

            # Calculate the direction vector of the antenna
            antenna_dir_x = end_x - start_x
            antenna_dir_y = end_y - start_y

            # Calculate the positions of the two cilia on each antenna
            for i in range(2):
                cilium_pos = (i + 1) / 2
                cilium_x = start_x + antenna_dir_x * cilium_pos
                cilium_y = start_y + antenna_dir_y * cilium_pos

                self.cilia.append((cilium_x, cilium_y))
        
        return self.cilia

    def update(self, x, y, angle):
        self.get_antennae(x, y,angle)
        self.get_cilia(x, y)


