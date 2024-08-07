import numpy as np
import mlx.core as mx

class SimulationConfig:
    def __init__(self, trial_number):

        # Target settings
        self.target_radius = 20
        self.target_hitbox = 40
        self.source_x = 1000
        self.source_y = 0

        # Arena settings
        self.arena_size_x = 1400
        self.arena_size_y = 1000

        # Wind settings
        self.wind_speed = 160
        self.wind_direction = 180 * np.pi / 180
        self.wind_switch_rate = 50
        self.initial_wind_steps = 5

        # Puff settings
        self.puff_birth_rate = 0.25
        self.puff_init_radius = 0.5
        self.puff_init_concentration = 10000
        self.diffusion_rate = 0.2
        self.random_walk_scale = 3.5
        self.max_puffs = 2500

        # Time settings
        self.time_step = 0.01
        self.max_time = 10

        # Entity settings
        self.stalled_time = 500
        self.max_outside_steps = 50

    @classmethod
    def get_instance(cls, trial_number=0):
        if not hasattr(cls, '_instance'):
            cls._instance = cls(trial_number)
        return cls._instance