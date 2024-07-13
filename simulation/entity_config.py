import numpy as np

class EntityConfig:
    def __init__(self, trial_number):
        self.arena_size_x = 1400
        self.arena_size_y = 1000

        self.delay = 250
        
        # Determine which scenario we're in based on the trial number
        # Trial 0 and even-numbered trials will have source_top_right as True
        self.source_top_right = trial_number % 2 == 0
        
        # Set source/target position
        if trial_number == 0:
            self.source_x = np.random.uniform(self.arena_size_x + 100, self.arena_size_x + 200)
            self.source_y = np.random.uniform(self.arena_size_y + 100, self.arena_size_y + 200)
        elif trial_number == 1:
            self.source_x = np.random.uniform(self.arena_size_x + 100, self.arena_size_x + 200)
            self.source_y = np.random.uniform(-200, -100)
        elif trial_number == 2:
            self.source_x = np.random.uniform(self.arena_size_x + 100, self.arena_size_x + 200)
            self.source_y = np.random.uniform(self.arena_size_y + 100, self.arena_size_y + 200)
        elif trial_number == 3:
            self.source_x = np.random.uniform(self.arena_size_x + 100, self.arena_size_x + 200)
            self.source_y = np.random.uniform(-200, -100)
            
        
        self.target_x = self.source_x
        self.target_y = self.source_y
        
        # Set entity position and angle
        if trial_number == 0:
            self.initial_x = np.random.uniform(150, 250)
            self.initial_y = np.random.uniform(100, 200)
            base_angle = np.pi / 2  # Facing up
        elif trial_number == 1:
            self.initial_x = np.random.uniform(150, 250)
            self.initial_y = np.random.uniform(800, 900)
            base_angle = -np.pi / 2  # Facing down
        elif trial_number == 2:
            self.initial_x = np.random.uniform(150, 250)
            self.initial_y = np.random.uniform(350, 450)
            base_angle = np.pi / 2  # Facing up
        elif trial_number == 3:
            self.initial_x = np.random.uniform(150, 250)
            self.initial_y = np.random.uniform(550, 650)
            base_angle = -np.pi / 2  # Facing down

        
        self.initial_angle = base_angle

        # Movement parameters
        self.speed = 3
        self.turn_angle = 3  # in degrees
        self.max_speed = 5
        self.max_turn_rate = 30

        self.sensor_count = 10  # number of receptors
        self.action_space = 3  # left, right, forward

        # Reward parameters
        self.target_threshold = 100  # Distance to consider "close" to the target
        self.environment_width = self.arena_size_x
        self.environment_height = self.arena_size_y