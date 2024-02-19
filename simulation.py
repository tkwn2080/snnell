import numpy as np
from collections import deque
import pygame
import sys
import time
import csv

from entity import OlfactoryEntity
from snn import Network

def collate_genotype(genotype):
    genotype_str = f"Learning Rate: {genotype[8][0]}, Eligibility Decay: {genotype[8][1]}, Recurrent Layer: {genotype[9]}"
    weights_str = genotype[5]
    return genotype_str, weights_str

class SimpleClock:
    def __init__(self, fps):
        self.start_time = time.time()  # Capture the start time when the instance is created
        self.last_tick = self.start_time
        self.fps = fps

    def tick(self):
        now = time.time()
        expected_tick = 1.0 / self.fps
        sleep_time = self.last_tick + expected_tick - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_tick = time.time()

    def get_ticks(self):
        # Calculate the elapsed time in milliseconds
        elapsed_time = (time.time() - self.start_time) * 1000  # Convert seconds to milliseconds
        return int(elapsed_time)

reward_signal = 0
distance_records = {}

individual_data = {}


def calculate_distance(emitter_x, emitter_y, final_x, final_y, individual_name):
    # Calculate the actual distance between the emitter and the final position
    actual_distance = np.hypot(final_x - emitter_x, final_y - emitter_y)
    
    # Calculate the maximum possible distance in the simulation space
    max_distance = np.hypot(1200, 800)
    
    # Normalize the actual distance to a 1-10 scale
    # The scale is inverted since a lower score means closer
    scaled_distance = (actual_distance / max_distance * 9) + 1
    
    # Ensure the scaled distance is within the bounds of 1-10
    scaled_distance = max(min(scaled_distance, 10), 1)
    
    # Check if this is the first record for this individual or if it's a new minimum distance
    # Only note down a record if the score is below 2.5
    if (individual_name not in distance_records) or (scaled_distance < 2.5 and scaled_distance < distance_records[individual_name]):
        distance_records[individual_name] = scaled_distance
        is_record = True
    else:
        is_record = False
    
    return scaled_distance, is_record

def run_simulation(genotype, current_trial, total_trials, current_candidate, total_candidates, current_epoch, num_epochs, network, individual_name, headless):
    if not headless:
        pygame.init()
        screen = pygame.display.set_mode((1200, 800), pygame.DOUBLEBUF | pygame.HWSURFACE)
        clock = pygame.time.Clock()
        print('HEAD')

    clock = SimpleClock(180) if headless else clock

    




    global reward_signal
    network = network

    white_particle_rate = 2  # How many white particles to emit per frame
    red_particle_rate = 1  # How many red particles to emit per frame once started

    length = genotype[0]
    probe_angle = genotype[1]
    response_angle = genotype[2]
    distance = genotype[3]
    speed = genotype[4]

    weights = genotype[5]
    architecture = genotype[6]
    depth = genotype[7]
    parameters = genotype[8]
    recurrence = genotype[9]

    # Particle properties using deque for efficient removal
    red_particles = deque()  # Red particles emitted from the top right corner

    white_particles = deque()  # White flow particles emitted from the right edge

    particle_radius = 2
    red_particle_emission_rate = red_particle_rate  # How many red particles to emit per frame once started

    # Timer setup for delayed red particle emission
    red_particle_emission_delay = 0  # Delay in milliseconds (2000ms = 2s)
    red_particle_emission_start_time = None  # Start time will be set after 2 seconds


    # Spatial partitioning setup
    grid_size = 50  # Define the size of the grid cells
    if not headless:
        num_grid_rows = screen.get_height() // grid_size
        num_grid_cols = screen.get_width() // grid_size
    else:
        num_grid_rows = 800 // grid_size
        num_grid_cols = 1200 // grid_size
    grid = [[deque() for _ in range(num_grid_cols)] for _ in range(num_grid_rows)]  # Grid initialized with deques


    # PARTICLE PHYSICS
    def check_collisions_in_grid(particle, grid):
        grid_x, grid_y = get_grid_cell(particle[0], particle[1])
        # Check the current cell and adjacent cells for collisions
        for y in range(max(0, grid_y - 1), min(num_grid_rows, grid_y + 2)):
            for x in range(max(0, grid_x - 1), min(num_grid_cols, grid_x + 2)):
                for other_particle in grid[y][x]:
                    if other_particle is not particle and check_collision(particle, other_particle):
                        # Simple bounce response by reversing velocities
                        particle[2], other_particle[2] = other_particle[2], particle[2]  # Exchange x velocities
                        particle[3], other_particle[3] = other_particle[3], particle[3]  # Exchange y velocities



    def check_collision(p1, p2):
        # Calculate distance between two particles
        dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
        return dist < particle_radius * 2

    def get_grid_cell(x, y):
        grid_x = min(max(int(x // grid_size), 0), num_grid_cols - 1)
        grid_y = min(max(int(y // grid_size), 0), num_grid_rows - 1)
        return grid_x, grid_y

    def add_to_grid(particle, grid):
        grid_x, grid_y = get_grid_cell(particle[0], particle[1])
        # No need for additional bounds check here due to get_grid_cell adjustment
        grid[grid_y][grid_x].append(particle)

    def remove_from_grid(particle, grid):
        grid_x, grid_y = get_grid_cell(particle[0], particle[1])
        # Try-except block to handle potential issues, though they should be mitigated now
        try:
            grid[grid_y][grid_x].remove(particle)
        except (IndexError, ValueError):
            pass  # Fail silently if the particle is not in the expected grid cell



    def update_particle(particle, grid):
        # Remove particle from its current grid cell
        remove_from_grid(particle, grid)
        # Update particle position, return False if out of bounds
        particle[0] += particle[2]
        particle[1] += particle[3]
        if 0 <= particle[0] < 1200 and 0 <= particle[1] < 800:
            # Add particle to its new grid cell
            add_to_grid(particle, grid)
            return False  # Keep in the list
        return True  # Mark for deletion

    def emit_white_particles(number_to_emit):
        for _ in range(number_to_emit):  # Emit 'number_to_emit' particles per frame
            white_particles.append([1200, np.random.randint(0, 800), -np.random.rand() * 2 - 0.5, np.random.rand() * 2 - 1])

    # Define the emitter's position and the larger sphere's radius
    # emitter_x = 900  # Keep the towards the right side
    # emitter_y = 200  # Vertical middle of the screen
    emitter_x = np.random.randint(900, 1100)  # Randomly vary the x-coordinate
    emitter_y = np.random.randint(200, 600)  # Randomly vary the y-coordinate
    emitter_radius = 10  # Radius of the larger sphere

    emitter = [emitter_x, emitter_y]

    def check_collision_with_emitter(entity_x, entity_y, emitter_x, emitter_y, entity_size, emitter_radius, probe_length, probe_angle):
        # Check collision between entity and emitter
        entity_emitter_distance = np.hypot(entity_x - emitter_x, entity_y - emitter_y)
        entity_emitter_collision = entity_emitter_distance < (entity_size + emitter_radius)
        
        # Check collision between probe and emitter
        probe_end_x = entity_x + np.cos(np.radians(probe_angle)) * probe_length
        probe_end_y = entity_y + np.sin(np.radians(probe_angle)) * probe_length
        probe_emitter_distance = np.hypot(probe_end_x - emitter_x, probe_end_y - emitter_y)
        probe_emitter_collision = probe_emitter_distance < (emitter_radius)
        
        return entity_emitter_collision or probe_emitter_collision


    def emit_red_particles():
        # Emit red particles from around the circumference of the larger sphere
        angle = np.random.rand() * 2 * np.pi  # Random angle for emission
        velocity_magnitude = 0.25  # Reduce the overall speed for a softer movement
        velocity_angle = np.random.rand() * 2 * np.pi  # Random direction for the velocity
        new_particle = [
            emitter_x + np.cos(angle) * emitter_radius,  # x-coordinate on the circumference
            emitter_y + np.sin(angle) * emitter_radius,  # y-coordinate on the circumference
            np.cos(velocity_angle) * velocity_magnitude,  # Initial x velocity
            np.sin(velocity_angle) * velocity_magnitude  # Initial y velocity
        ]
        red_particles.append(new_particle)
        add_to_grid(new_particle, grid)  # Add the new particle to the grid

    simulation_start_time = clock.get_ticks()  # Get the start time of the simulation
    simulation_time_limit = 45000  # Set a time limit for the simulation (in milliseconds)
    opening_countdown = 0  # Countdown to the opening of the prongs

    olfactory_entity = OlfactoryEntity(100, 400, length, probe_angle, response_angle, distance, speed, network)
    entity_start_time = 2000

    first_draw = True

    # Simulation loop for the current genotype
    while True:
        current_time = clock.get_ticks()
        if current_time - simulation_start_time > simulation_time_limit:
            
            # Punish the entity for not finding the emitter in time

            distance, is_record = calculate_distance(emitter_x, emitter_y, olfactory_entity.x, olfactory_entity.y, individual_name)
            if is_record == True:
                reward_signal += 50
                print(f'Reward for closest distance: {distance}.')
            else: 
                if olfactory_entity.particle_count != 0:
                    punishment = (100 * distance / (olfactory_entity.particle_count / 10))
                else:
                    punishment = 100 * distance
                reward_signal -= punishment
                print(f'Punishment for not finding the emitter in time: {punishment}. Ending simulation.')
            Network.modify_learning(network, reward_signal)

            break  # Exit the simulation loop after the time limit

        



        # Simulation boundaries
        BOUNDARY_LEFT = 0
        BOUNDARY_RIGHT = 1200
        BOUNDARY_TOP = 0
        BOUNDARY_BOTTOM = 800

        if olfactory_entity.x < BOUNDARY_LEFT or olfactory_entity.x > BOUNDARY_RIGHT or olfactory_entity.y < BOUNDARY_TOP or olfactory_entity.y > BOUNDARY_BOTTOM:
            
            distance, is_record = calculate_distance(emitter_x, emitter_y, olfactory_entity.x, olfactory_entity.y, individual_name)
            if is_record == True:
                reward_signal += 50
                print(f'Reward for closest distance: {distance}.')
            else: 
                if olfactory_entity.particle_count != 0:
                    punishment = (100 * distance / (olfactory_entity.particle_count / 10))
                else:
                    punishment = 100 * distance
                print(f"Punishment for going out of bounds: {punishment}. Ending simulation.")
                reward_signal -= punishment
            Network.modify_learning(network, reward_signal)
            return {
                'collided': False,
                'collision_time': None,
                'final_position': (olfactory_entity.x, olfactory_entity.y),
                'simulation_time': clock.get_ticks() - simulation_start_time,  # Total simulation time
                'emitter_position': (emitter_x, emitter_y),
                'emitter_radius': emitter_radius,
                'particle_count': olfactory_entity.particle_count
            }

        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         filename = time.strftime("%d%m%Y%H") + '_individual_data.csv'
        #         csv_filename = './records/' + filename
        #         with open(csv_filename, 'w', newline='') as csvfile:
        #             fieldnames = ['name', 'genotype', 'fitness', 'heritage', 'epoch', 'genotype', 'weights']
        #             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        #             writer.writeheader()
        #             for individual_name, data in individual_data.items():
        #                 collated_genotype, collated_weights = collate_genotype(data['genotype'])
        #                 writer.writerow({
        #                     'name': individual_name,
        #                     'fitness': data['fitness'],
        #                     'heritage': data['heritage'],
        #                     'epoch': data['epoch'],
        #                     'genotype': collated_genotype,
        #                     'weights': collated_weights
        #                 })
        #         print(f"Individual data exported to {csv_filename}.")
        #         pygame.quit()
        #         sys.exit()

        # Simulation logic (particle movement, drawing, etc.) goes here
        # Make sure to update the screen with pygame.display.flip() and limit the frame rate with clock.tick()
    
        # Clear screen
        if not headless:
            screen.fill((0, 0, 0))

        # Reset grid for spatial partitioning
        grid = [[deque() for _ in range(num_grid_cols)] for _ in range(num_grid_rows)]

        # Draw the larger red sphere from which particles are emitted
        if not headless:
            pygame.draw.circle(screen, (200, 0, 0), (emitter_x, emitter_y), emitter_radius)

        # Emit white particles
        emit_white_particles(white_particle_rate)  # Emit n white particles per frame

        # Get current time
        current_time = clock.get_ticks()

        # If the emission start time is not set and the delay has passed, start emitting red particles
        if red_particle_emission_start_time is None and current_time > red_particle_emission_delay:
            red_particle_emission_start_time = current_time  # Set the start time for red particle emission

        # If the red particle emission has started, emit particles based on the rate with a 50% probability
        if red_particle_emission_start_time is not None:
            for _ in range(red_particle_emission_rate):
                if np.random.rand() < 0.2:  # 20% chance to emit a particle
                    emit_red_particles()

        # Check for collision with the red emitter
        if not olfactory_entity.collided_with_emitter:  # Check only if collision hasn't occurred yet
            if check_collision_with_emitter(olfactory_entity.x, olfactory_entity.y, emitter_x, emitter_y, olfactory_entity.size, emitter_radius, olfactory_entity.prong_length, olfactory_entity.prong_angle):
                olfactory_entity.collided_with_emitter = True
                olfactory_entity.collision_time = clock.get_ticks() - simulation_start_time
                
                # Reward the entity for finding the emitter
                reward_signal += 100
                Network.modify_learning(network, reward_signal)
                
                print("Collision with emitter at time:", olfactory_entity.collision_time)
                return {
                    'collided': False,
                    'collision_time': None,
                    'final_position': (olfactory_entity.x, olfactory_entity.y),
                    'simulation_time': clock.get_ticks() - simulation_start_time,  # Total simulation time
                    'emitter_position': (emitter_x, emitter_y),
                    'emitter_radius': emitter_radius,
                    'particle_count': olfactory_entity.particle_count
                }

        opening_countdown += 1

        if opening_countdown > 600:
            if olfactory_entity.timesteps_since_touch > 1200:

                distance, is_record = calculate_distance(emitter_x, emitter_y, olfactory_entity.x, olfactory_entity.y, individual_name)
                if is_record == True:
                    reward_signal += 50
                    print(f'Reward for closest distance: {distance}.')
                else: 
                    if olfactory_entity.particle_count != 0:
                        punishment = (100 * distance / (olfactory_entity.particle_count / 10))
                    else:
                        punishment = 100 * distance
                    reward_signal -= punishment
                    print(f"Punishment for getting lost: {punishment}. Ending simulation.")

                return {
                    'collided': False,
                    'collision_time': None,
                    'final_position': (olfactory_entity.x, olfactory_entity.y),
                    'simulation_time': clock.get_ticks() - simulation_start_time,  # Total simulation time
                    'emitter_position': (emitter_x, emitter_y),
                    'emitter_radius': emitter_radius,
                    'particle_count': olfactory_entity.particle_count
                }
        
        olfactory_entity.timesteps_since_touch += 1
        

        for particle in list(white_particles):
            if update_particle(particle, grid):
                white_particles.remove(particle)

        for particle in list(red_particles):
            if update_particle(particle, grid):  # This function already handles grid updates
                red_particles.remove(particle)
            else:
                check_collisions_in_grid(particle, grid)
                    # New function to check collisions efficiently

        # Update and draw white particles
        temp_white_particles = deque()  # Temporary deque to hold updated particles
        for particle in white_particles:
            if not update_particle(particle, grid):  # Now passing 'grid' as an argument
                temp_white_particles.append(particle)
                # Draw the particle after confirming it's still within bounds
                if not headless:
                    pygame.draw.circle(screen, (255, 255, 255), (int(particle[0]), int(particle[1])), particle_radius)
        white_particles = temp_white_particles  # Replace the original deque with the updated one

        # Similarly, update and draw red particles
        temp_red_particles = deque()  # Temporary deque to hold updated particles
        for particle in red_particles:
            if not update_particle(particle, grid):  # Passing 'grid' as an argument
                temp_red_particles.append(particle)
                # Draw the particle after confirming it's still within bounds
                if not headless:
                    pygame.draw.circle(screen, (255, 0, 0), (int(particle[0]), int(particle[1])), particle_radius)
        red_particles = temp_red_particles  # Replace the original deque with the updated one




        if current_time - simulation_start_time > entity_start_time:
            if first_draw == True:
                if not headless:
                    olfactory_entity.draw(screen)
                first_draw = False
            olfactory_entity.update(list(red_particles), emitter)  # Pass a list of red_particles for sensing
            if not headless:
                olfactory_entity.draw(screen)

        if reward_signal != 0:
            Network.modify_learning(network, reward_signal)
            reward_signal = 0

        # Display trial, candidate number, epoch number, architecture, learning rate, and decay rate
        if not headless:
            font = pygame.font.Font(None, 36)
            trial_text = f"Trial: {current_trial}/{total_trials}"
            candidate_text = f"Candidate: {current_candidate}/{total_candidates}"
            epoch_text = f"Epoch: {current_epoch}/{num_epochs}"
            architecture_text = f"Architecture: {architecture}"
            learning_rate_text = f"Learning Rate: {parameters[0]}"
            decay_rate_text = f"Eligibility Decay: {parameters[1]}"
            recurrent_layer_text = f"Recurrent Layer: {recurrence}"
            trial_surface = font.render(trial_text, True, (255, 255, 255))
            candidate_surface = font.render(candidate_text, True, (255, 255, 255))
            epoch_surface = font.render(epoch_text, True, (255, 255, 255))
            architecture_surface = font.render(architecture_text, True, (255, 255, 255))
            learning_rate_surface = font.render(learning_rate_text, True, (255, 255, 255))
            decay_rate_surface = font.render(decay_rate_text, True, (255, 255, 255))
            recurrent_layer_surface = font.render(recurrent_layer_text, True, (255, 255, 255))
            screen.blit(trial_surface, (10, screen.get_height() - 210))
            screen.blit(candidate_surface, (10, screen.get_height() - 180))
            screen.blit(epoch_surface, (10, screen.get_height() - 150))
            screen.blit(architecture_surface, (10, screen.get_height() - 120))
            screen.blit(learning_rate_surface, (10, screen.get_height() - 90))
            screen.blit(decay_rate_surface, (10, screen.get_height() - 60))
            screen.blit(recurrent_layer_surface, (10, screen.get_height() - 30))

            pygame.display.flip()
            clock.tick(180)  # Limit to 60 frames per second

    return {
        'collided': olfactory_entity.collided_with_emitter,
        'collision_time': olfactory_entity.collision_time if olfactory_entity.collided_with_emitter else None,
        'final_position': (olfactory_entity.x, olfactory_entity.y),
        'simulation_time': clock.get_ticks() - simulation_start_time,  # Total simulation time
        'emitter_position': (emitter_x, emitter_y),
        'emitter_radius': emitter_radius,
        'particle_count': olfactory_entity.particle_count
    }