import numpy as np
import mlx.core as mx
import time
import pygame

from simulation.environment import WindSystem, PuffSystem
from simulation.sim_config import SimulationConfig
from simulation.entity import Entity
from handler import Handler

class Simulation:
    def __init__(self, config: SimulationConfig, handler: Handler):
        self.config = config
        self.handler = handler
        self.wind = WindSystem(config)
        self.puffs = PuffSystem(config)

        self.current_time = 0
        self.collision = False
        self.collision_time = None
        self.stalled_counter = 0
        self.last_position = None
        self.record_time = 2
        self.record_data = []

        self.entity = Entity()
        # self.entity.config.initial_x = np.random.uniform(50, 200)
        # self.entity.config.initial_y = self.config.source_y + (450 if self.config.source_y < 0 else 350) + np.random.uniform(-50, 50)
        # self.entity.config.initial_angle = np.random.uniform(0, 2 * np.pi)
        self.entity.handler = self.handler
        self.entity.target = [self.config.source_x, self.config.source_y]

        self.origin_x = self.entity.config.initial_x
        self.origin_y = self.entity.config.initial_y

        self.start_time = time.time()
        self.out_of_bounds_counter = 0
        self.max_out_of_bounds = 50  # Allow 50 steps out of bounds before terminating
        self.is_out_of_bounds = False
        self.is_stalled = False

    def update(self):
        self.wind.update(self.current_time, 'constant')
        
        num_new_puffs = np.random.poisson(self.config.puff_birth_rate)
        self.puffs.activate_puffs(num_new_puffs, self.current_time)
        self.puffs.update(self.config.time_step, self.wind)
        
        # Handle environmental checks and updates
        self.check_collision()
        self.environment_update()

        self.current_time += self.config.time_step

    def final_call(self, cause):
        # Collect final state from environment
        environment = {
            'wind': self.wind,
            'puffs': self.puffs.puffs
        }

        final_state = self.entity.get_state(environment)

        # Calculate final reward
        final_reward = self.calculate_reward(self.collision, self.is_out_of_bounds)

        # Apply additional punishment based on termination cause
        if cause == 'out_of_bounds':
            final_reward -= 50  # Additional punishment for being out of bounds
        elif cause == 'stalled':
            final_reward -= 25  # Punishment for being stalled

        # Update the handler with the final state, reward, and done flag
        self.handler.update(final_reward, final_state, done=True)

        # Perform a final update on the entity (if needed)
        self.entity.update(environment)

    def calculate_reward(self, collision, is_out_of_bounds):
        if collision:
            return 100  # Large positive reward for reaching the target
        elif is_out_of_bounds:
            return -50  # Negative reward for being out of bounds

    def environment_update(self):
        is_out_of_bounds = self.is_it_out_of_bounds()
        if is_out_of_bounds:
            self.out_of_bounds_counter += 1
        else:
            self.out_of_bounds_counter = max(0, self.out_of_bounds_counter - 1)  # Decrease counter if in bounds

        environment = {
            'wind': self.wind,
            'puffs': self.puffs.puffs
        }

        boundary_punishment = 0.5 * self.out_of_bounds_counter

        self.handler.update_environment(environment, boundary_punishment)

        self.entity.update(environment)

    def check_collision(self):
        entity_x, entity_y, _ = self.entity.get_position()
        normalised_entity_y = entity_y - 500
        entity_emitter_distance = np.hypot(entity_x - self.config.source_x, normalised_entity_y - self.config.source_y)

        collision_distance = self.config.target_radius + self.config.target_hitbox

        if entity_emitter_distance < (collision_distance):  # entity_size + emitter_radius
            self.collision = True
            self.collision_time = self.current_time

    def check_stalled(self):
        entity_x, entity_y, _ = self.entity.get_position()
        current_position = (entity_x, entity_y)
        if self.last_position is None:
            self.last_position = current_position
        elif self.last_position == current_position:
            self.stalled_counter += 1
        else:
            self.stalled_counter = 0
            self.last_position = current_position

    def sample_behaviour(self):
        if self.current_time >= self.record_time:
            self.record_time += 1
            sample = self.entity.get_position()
            standardized_x = sample[0] - self.origin_x
            standardized_y = sample[1] - self.origin_y
            standardized_sample = [standardized_x, standardized_y] + list(sample[2:])
            self.record_data.append(standardized_sample)

    def run(self, headless=True, environment=None):
        if not headless:
            screen, clock = environment
            pygame.display.set_caption("snnell")

        while self.current_time < self.config.max_time:
            if not headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print('Exiting simulation via QUIT event')
                        self.final_call('quit')
                        return self.get_results()
                
                screen.fill((0, 0, 0))
                self.draw_puffs(screen)
                self.entity.draw(screen)
                pygame.display.flip()

            self.update()

            if self.out_of_bounds_counter >= self.max_out_of_bounds:
                self.is_out_of_bounds = True
                self.final_call('out_of_bounds')
                return self.get_results()

            if self.stalled_counter > self.config.stalled_time:
                self.is_stalled = True
                self.final_call('stalled')
                return self.get_results()

            if self.collision:
                self.final_call('collision')
                return self.get_results()

        # If we've reached the maximum time without termination
        self.final_call('time_limit')
        return self.get_results()

    def is_it_out_of_bounds(self):
        entity_x, entity_y, _ = self.entity.get_position()
        if (entity_x < 0 or entity_x > self.config.arena_size_x or 
                entity_y < 0 or entity_y > self.config.arena_size_y):
            return True
        else:
            return False

    def get_results(self):
        self.handler.done()
        return {
            'collided': self.collision,
            'collision_time': self.collision_time,
            'simulation_time': time.time() - self.start_time,
            'final_position': self.entity.get_position()[:2],
            'emitter_position': (self.config.source_x, self.config.source_y),
            'behaviour_record': self.get_behaviour_record(),
        }

    def get_behaviour_record(self):
        if len(self.record_data) < 15:
            for _ in range(15 - len(self.record_data)):
                record = self.entity.get_position()
                standardized_x = record[0] - self.origin_x
                standardized_y = record[1] - self.origin_y
                standardized_record = [standardized_x, standardized_y] + list(record[2:])
                self.record_data.append(standardized_record)
        return self.record_data[:15]

    def draw_puffs(self, screen):
        active_mask = self.puffs.puffs['time'] >= 0
        numeric_mask = active_mask.astype(mx.int32)

        x_positions = (self.puffs.puffs['x'] * numeric_mask).tolist()
        y_positions = ((self.puffs.puffs['y'] + self.config.arena_size_y / 2) * numeric_mask).tolist()
        radii = (self.puffs.puffs['radius'] * numeric_mask).tolist()
        concentrations = (self.puffs.puffs['concentration'] * numeric_mask).tolist()

        for x, y, radius, concentration in zip(x_positions, y_positions, radii, concentrations):
            if radius > 0:
                puff_color = (255, 255, 255)
                puff_radius = int(radius * 100)
                puff_pos = (int(x), int(y))
                pygame.draw.circle(screen, puff_color, puff_pos, puff_radius)

        source_color = (255, 0, 0)
        source_pos = (self.config.source_x, int(self.config.source_y + self.config.arena_size_y/2))
        pygame.draw.circle(screen, source_color, source_pos, self.config.target_radius)