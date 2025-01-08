import numpy as np

class EntityConfig:
    def __init__(self, trial_number, generation):
        self.arena_size_x = 1400
        self.arena_size_y = 1000
        self.delay = 220
        self.generation = generation + 1

        # Cap the generation at 100 for variability calculations
        capped_generation = min(self.generation, 100)

        # Increase variability based on generation, capped at 100
        self.variability = min(capped_generation, 100)
        
        # Determine which scenario we're in based on the trial number
        self.source_top_right = trial_number % 2 == 0
        
        # Set source/target position with balanced variability
        if self.source_top_right:
            self.source_x = np.random.uniform(self.arena_size_x + 100 - self.variability, self.arena_size_x + 200 + self.variability)
            self.source_y = np.random.uniform(self.arena_size_y + 100 - self.variability, self.arena_size_y + 200 + self.variability)
        else:
            self.source_x = np.random.uniform(self.arena_size_x + 100 - self.variability, self.arena_size_x + 200 + self.variability)
            self.source_y = np.random.uniform(-200 - self.variability, -100 + self.variability)
        
        self.target_x = self.source_x
        self.target_y = self.source_y
        
        # Set entity position and angle with balanced variability
        # Adjust the ranges to ensure entities start well within the arena
        safe_margin = 150  # Increased margin for safety
        trial_mod_4 = trial_number % 4
        if trial_mod_4 == 0:
            self.initial_x = np.random.uniform(safe_margin, safe_margin + 100)
            self.initial_y = np.random.uniform(safe_margin, safe_margin + 100)
            base_angle = np.pi / 2  # Facing up
        elif trial_mod_4 == 1:
            self.initial_x = np.random.uniform(safe_margin, safe_margin + 100)
            self.initial_y = np.random.uniform(self.arena_size_y - safe_margin - 100, self.arena_size_y - safe_margin)
            base_angle = -np.pi / 2  # Facing down
        elif trial_mod_4 == 2:
            self.initial_x = np.random.uniform(safe_margin, safe_margin + 100)
            self.initial_y = np.random.uniform(self.arena_size_y / 2 - 50, self.arena_size_y / 2 + 50)
            base_angle = np.pi / 2  # Facing up
        else:  # trial_mod_4 == 3
            self.initial_x = np.random.uniform(safe_margin, safe_margin + 100)
            self.initial_y = np.random.uniform(self.arena_size_y * 3/4 - 50, self.arena_size_y * 3/4 + 50)
            base_angle = -np.pi / 2  # Facing down

        # Add variability to the initial position, ensuring it stays within bounds
        max_pos_variability = min(self.variability, safe_margin - 50)  # Limit variability to avoid edge cases
        self.initial_x += np.random.uniform(-max_pos_variability, max_pos_variability)
        self.initial_y += np.random.uniform(-max_pos_variability, max_pos_variability)

        # Ensure the initial position is within the arena bounds
        self.initial_x = np.clip(self.initial_x, safe_margin, self.arena_size_x - safe_margin)
        self.initial_y = np.clip(self.initial_y, safe_margin, self.arena_size_y - safe_margin)

        # Calculate angle variability (max 45 degrees)
        max_angle_variability = np.radians(45)
        current_angle_variability = (capped_generation / 100) * max_angle_variability

        # Add variability to the initial angle
        self.initial_angle = base_angle + np.random.uniform(-current_angle_variability, current_angle_variability)

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