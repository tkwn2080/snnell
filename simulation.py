import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numpy as np
import time
import mlx.core as mx
from entity import Entity
from snn import Network
from evolve import Population, Individual

class Simulation:
    def __init__(self, emitter_x, emitter_y, neuron_type, recurrent=False):
        self.arena_size_x = 1200
        self.arena_size_y = 800
        self.wind_speed = 140
        self.wind_direction = mx.array(180 * mx.pi / 180)
        self.puff_birth_rate = 1.0
        self.puff_init_radius = 0.05
        self.puff_init_concentration = 100
        self.diffusion_rate = 0.4
        self.random_walk_scale = 3
        self.time_step = 0.01
        self.max_time = 15

        # COLLISION
        self.collision = False
        self.collision_time = None

        # SOURCE
        self.source_x = emitter_x 
        self.source_y = emitter_y 

        # WIND
        self.initial_wind_steps = 5
        self.wind_switch_rate = 50
        self.wind_switch_count = 0

        if emitter_y < 0:
            self.wind_switch = True
        elif emitter_y > 0:
            self.wind_switch = False

        # ENTITY
        self.entity_x = np.random.uniform(50,200)
        if emitter_y < 0:
            self.entity_y = emitter_y + 450 + np.random.uniform(0,100)
        elif emitter_y > 0:
            self.entity_y = emitter_y + 350 - np.random.uniform(0,100)

        # BEHAVIOUR
        self.angle = None
        self.record_time = 2
        self.record_data = []
        self.origin_x = self.entity_x
        self.origin_y = self.entity_y
        
        # NETWORK
        self.neuron_type = neuron_type
        self.recurrent = recurrent
        
        # Initialize puffs dataframe
        # self.puffs_df = pd.DataFrame(columns=['puff_number', 'time', 'x', 'y', 'radius', 'concentration'])
        
        # Initialize centerline dataframe
        # self.centerline_df = pd.DataFrame(columns=['puff_number', 'time', 'x', 'y'])

        self.start_time = time.time()

        # MLX IMPLEMENTATION
        self.max_puffs = 5000
        self.active_puffs = 0

        self.puffs = {
            'time': mx.full(self.max_puffs, -1.0),
            'x': mx.full(self.max_puffs, self.source_x),
            'y': mx.full(self.max_puffs, self.source_y),
            'radius': mx.full(self.max_puffs, self.puff_init_radius),
            'concentration': mx.full(self.max_puffs, self.puff_init_concentration)
        }

    def activate_puffs(self, n_puffs, time):
        start_index = self.active_puffs
        end_index = start_index + n_puffs

        if end_index <= len(self.puffs['time']):
            self.puffs['time'][start_index:end_index] = time
            self.active_puffs += n_puffs
        else:
            print('Too many puffs!')
        
    def update_wind(self, wind_config):
        if wind_config == 'constant':
            pass

        elif wind_config == 'switch-once':
            if self.wind_switch_count < 1:
                if self.current_time >= self.initial_wind_steps:
                    self.wind_direction += 10 * mx.pi / 180  
                    self.wind_switch_count += 1
                    print('SWITCHING')

        elif wind_config == 'switch-many':
            if self.current_time >= self.initial_wind_steps:
                self.wind_switch_count += 1
                if self.wind_switch_count >= self.wind_switch_rate:
                    if self.wind_switch == False:
                        self.wind_direction += 10 * mx.pi / 180
                        self.wind_switch = True
                        self.wind_switch_count = 0
                    elif self.wind_switch == True:
                        self.wind_direction -= 10 * mx.pi / 180
                        self.wind_switch = False
                        self.wind_switch_count = 0
    
    def update_puffs(self):

        # Only apply updates if the puff is active
        active_mask = self.puffs['time'] >= 0
        numeric_mask = active_mask.astype(mx.float32) # MLX says Booleans are not supported "yet"

        # Add random walk to the puff updates
        rand_mask = mx.random.uniform(low=0, high=2, shape=[self.max_puffs]) * self.random_walk_scale
    
        # Eventually this will crash if the angle bends all the way back?
        dx = self.wind_speed * mx.cos(self.wind_direction) * self.time_step
        dy = self.wind_speed * mx.sin(self.wind_direction) * self.time_step

        self.puffs['x'] += dx * numeric_mask * (rand_mask / 2)
        self.puffs['y'] += dy * numeric_mask * (rand_mask * 2)

        radius_increase = self.diffusion_rate * self.time_step * mx.random.normal(shape=[self.max_puffs], loc=1, scale=0.2)
        self.puffs['radius'] += radius_increase * numeric_mask
        self.puffs['concentration'] = (self.puff_init_radius / self.puffs['radius']) ** 3 * self.puff_init_concentration * numeric_mask

    # def update_centerline(self):
    #     self.centerline_df['x'] += self.wind_speed * np.cos(np.radians(self.wind_direction)) * self.time_step
    #     self.centerline_df['y'] += self.wind_speed * np.sin(np.radians(self.wind_direction)) * self.time_step

    # def get_wind(self, x, y, wind_speed):
    #     # Calculate the wind velocity at the given position
    #     wind_x = wind_speed * mx.cos(self.wind_direction)
    #     wind_y = wind_speed * mx.sin(self.wind_direction)
    #     return wind_x, wind_y

    def draw_puffs(self, screen):
        active_mask = self.puffs['time'] >= 0
        numeric_mask = active_mask.astype(mx.int32)  # Convert boolean mask to int for multiplication

        # Use numeric_mask to filter out inactive puffs by setting their attributes to zero
        x_positions = (self.puffs['x'] * numeric_mask).tolist()
        y_positions = ((self.puffs['y'] + self.arena_size_y / 2) * numeric_mask).tolist()
        radii = (self.puffs['radius'] * numeric_mask).tolist()
        concentrations = (self.puffs['concentration'] * numeric_mask).tolist()

        for x, y, radius, concentration in zip(x_positions, y_positions, radii, concentrations):
            # print(f'Puff x: {x}, y: {y}, radius: {radius}, concentration: {concentration}')
            if radius > 0:  # Check if the puff is active based on concentration
                puff_color = (255, 255, 255, int(concentration * 255 / self.puff_init_concentration))
                puff_radius = int(radius * 100)  # Scale up the radius for visibility
                puff_pos = (int(x), int(y))
                pygame.draw.circle(screen, puff_color, puff_pos, puff_radius)
        
        # Draw centerline
        # centerline_color = (255, 0, 0)  # Red color for centerline
        # centerline_points = [(int(row['x']), int(row['y'] + self.arena_size_y/2)) for _, row in self.centerline_df.iterrows()]
        # if len(centerline_points) >= 2:
        #     pygame.draw.lines(screen, centerline_color, False, centerline_points)
        
        # Draw source point
        source_color = (255, 0, 0)  # Red color for source point
        source_pos = (self.source_x, int(self.source_y + self.arena_size_y/2))  # Adjust y-coordinate for PyGame
        pygame.draw.circle(screen, source_color, source_pos, 10)  # Draw a circle with radius 10
    
    def check_collision(self, entity, emitter_x, emitter_y):
        # Checks whether the entity has collided with the emitter
        normalised_entity_y = entity.y - 400
        entity_emitter_distance = np.hypot(entity.x - emitter_x, normalised_entity_y - emitter_y)
        entity_size = 12
        emitter_radius = 20
        if entity_emitter_distance < (entity_size + emitter_radius):
            return True
        else:
            return False

    def sample_behaviour(self, entity):
        if self.current_time >= self.record_time:
            self.record_time += 1
            sample = Entity.get_entity(entity)
            
            # Standardize the coordinates by subtracting the origin
            standardized_x = sample[0] - self.origin_x
            standardized_y = sample[1] - self.origin_y
            
            # Create a new standardized sample
            standardized_sample = [standardized_x, standardized_y] + list(sample[2:])
            
            self.record_data.append(standardized_sample)

    def behaviour_record(self, entity):
        if len(self.record_data) < 15:
            # Pad to 15 with the final position
            for _ in range(15 - len(self.record_data)):
                record = Entity.get_entity(entity)
                
                # Standardize the coordinates by subtracting the origin
                standardized_x = record[0] - self.origin_x
                standardized_y = record[1] - self.origin_y
                
                # Create a new standardized record
                standardized_record = [standardized_x, standardized_y] + list(record[2:])
                
                self.record_data.append(standardized_record)
            
            return self.record_data
        
        elif len(self.record_data) == 15:
            return self.record_data
        
        else:
            length = len(self.record_data)
            print(f'Length of record too long: {length}')

        
         
    def simulate(self, wind_config, emitter_x, emitter_y, individual, neuron_type, headless, recurrent, environment=None):
        self.current_time = 0
        puff_counter = 0
        outside_steps = 0  # Initialize outside_steps counter
        max_outside_steps = 50

        if not headless:
            screen, clock = environment
            pygame.display.set_caption("snnell")

        # INITIATE NETWORK AND ENTITY
        self.network = Network(individual, neuron_type, self.recurrent)
        entity = Entity(self.network, [self.entity_x, self.entity_y])
        
        # if not headless:
        #     pygame.init()
        #     screen = pygame.display.set_mode((self.arena_size_x, self.arena_size_y))
        #     pygame.display.set_caption("Plume Simulation")
        #     clock = pygame.time.Clock()
            
        while self.current_time < self.max_time:
            if not headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print('Exiting simulation via QUIT event')
                        return
                        
                screen.fill((0, 0, 0))
                
            # WIND
            self.update_wind(wind_config)

            # PUFFS
            num_new_puffs = np.random.poisson(self.puff_birth_rate)
            self.activate_puffs(num_new_puffs, self.current_time)
            self.update_puffs()

            entity.x, entity.y, entity.angle = Entity.get_entity(entity)
            state = [[self.wind_direction, self.wind_speed], self.puffs]
            
            Entity.update(entity, state)
            
            if not headless:
                self.draw_puffs(screen)
                Entity.draw(entity, screen)
                
                pygame.display.flip()  # Update the display
            
            self.current_time += self.time_step
            elapsed_time = time.time() - self.start_time

            # Check if the creature is outside the arena boundaries
            if entity.x < 0 or entity.x > self.arena_size_x or entity.y < 0 or entity.y > self.arena_size_y:
                outside_steps += 1
                if outside_steps > max_outside_steps:
                    return {
                        'collided': self.collision,
                        'collision_time': self.collision_time,
                        'simulation_time': elapsed_time,
                        'final_position': (entity.x, entity.y),
                        'emitter_position': (emitter_x, emitter_y),
                        'behaviour_record': self.behaviour_record(entity),
                    }
            else:
                outside_steps = 0  # Reset outside_steps counter if creature is inside the arena

            self.collision = self.check_collision(entity, emitter_x, emitter_y)
            if self.collision:
                print(f'Collision detected at time: {self.current_time}')
                return {
                    'collided': self.collision,
                    'collision_time': self.current_time,
                    'simulation_time': elapsed_time,
                    'final_position': (entity.x, entity.y),
                    'emitter_position': (emitter_x, emitter_y),
                    'behaviour_record': self.behaviour_record(entity),
                }
            
            self.sample_behaviour(entity)

        return {
            'collided': self.collision,
            'collision_time': self.collision_time,
            'simulation_time': elapsed_time,
            'final_position': (entity.x, entity.y),
            'emitter_position': (emitter_x, emitter_y),
            'behaviour_record': self.behaviour_record(entity),
        }

# # This is only for testing, otherwise run through index.py
# for i in range(10):
#     emitter_x = np.random.randint(900, 1100)
#     emitter_y = np.random.randint(-150, 150)
#     simulation = Simulation(emitter_x, emitter_y, 'standard')
#     simulation.simulate(wind_config='constant', headless=False)