import numpy as np

class EntityConfig:
    def __init__(self):
        self.initial_x = 300
        self.initial_y = 500

        self.target_x = 1000
        self.target_y = 0
        
        # Randomly generate an angle facing in a random direction
        # self.initial_angle = np.random.uniform(0, 2 * np.pi)

        self.initial_angle = 0

        self.speed = 5
        self.turn_angle = 5  # in degrees
        self.sensor_count = 10  # number of receptors
        self.action_space = 3  # left, right, forward
        self.arena_size_x = 1400
        self.arena_size_y = 1000