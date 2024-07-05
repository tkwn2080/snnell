import mlx.core as mx
import numpy as np

from simulation.sim_config import SimulationConfig

class WindSystem:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.direction = config.wind_direction
        self.speed = config.wind_speed
        self.switch_count = 0
        self.switch = config.source_y < 0

    def update(self, current_time, wind_config):
        if wind_config == 'constant':
            return

        if wind_config == 'switch-once':
            if self.switch_count < 1 and current_time >= self.config.initial_wind_steps:
                self.direction += 10 * np.pi / 180
                self.switch_count += 1
                print('SWITCHING')

        elif wind_config == 'switch-many':
            if current_time >= self.config.initial_wind_steps:
                self.switch_count += 1
                if self.switch_count >= self.config.wind_switch_rate:
                    if not self.switch:
                        self.direction += 10 * np.pi / 180
                        self.switch = True
                    else:
                        self.direction -= 10 * np.pi / 180
                        self.switch = False
                    self.switch_count = 0

class PuffSystem:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.active_puffs = 0
        self.puffs = {
            'time': mx.full(config.max_puffs, -1.0),
            'x': mx.full(config.max_puffs, config.source_x),
            'y': mx.full(config.max_puffs, config.source_y),
            'radius': mx.full(config.max_puffs, config.puff_init_radius),
            'concentration': mx.full(config.max_puffs, config.puff_init_concentration)
        }

    def activate_puffs(self, n_puffs, time):
        start_index = self.active_puffs
        end_index = start_index + n_puffs

        if end_index <= len(self.puffs['time']):
            self.puffs['time'][start_index:end_index] = time
            self.active_puffs += n_puffs
        else:
            print('Too many puffs!')

    def update(self, time_step, wind):
        active_mask = self.puffs['time'] >= 0
        numeric_mask = active_mask.astype(mx.float32)

        rand_mask = mx.random.uniform(low=0, high=2, shape=[self.config.max_puffs]) * self.config.random_walk_scale

        dx = wind.speed * mx.cos(wind.direction) * time_step
        dy = wind.speed * mx.sin(wind.direction) * time_step

        self.puffs['x'] += dx * numeric_mask * (rand_mask / 2)
        self.puffs['y'] += dy * numeric_mask * (rand_mask * 2)

        radius_increase = self.config.diffusion_rate * time_step * mx.random.normal(shape=[self.config.max_puffs], loc=1, scale=0.2)
        self.puffs['radius'] += radius_increase * numeric_mask
        self.puffs['concentration'] = (self.config.puff_init_radius / self.puffs['radius']) ** 3 * self.config.puff_init_concentration * numeric_mask

