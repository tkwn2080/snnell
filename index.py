import pygame
import sys
import numpy as np
from collections import deque
import random
import string
import csv

from evolve import init_population, reproduction
from entity import OlfactoryEntity
from snn import Network, Layer

pygame.init()
screen = pygame.display.set_mode((1200, 800), pygame.DOUBLEBUF | pygame.HWSURFACE)
clock = pygame.time.Clock()

reward_signal = 0

trial_csv_filename = 'trial_data.csv'
with open(trial_csv_filename, 'w', newline='') as trial_csvfile:
    trial_fieldnames = ['trial_number', 'individual_name', 'particle_count', 'fitness']
    trial_writer = csv.DictWriter(trial_csvfile, fieldnames=trial_fieldnames)
    trial_writer.writeheader()


def run_simulation(genotype, screen, clock, current_trial, total_trials, current_candidate, total_candidates, current_epoch, num_epochs, network):

    global reward_signal
    network = network

    white_particle_rate = 5  # How many white particles to emit per frame
    red_particle_rate = 3  # How many red particles to emit per frame once started

    length = genotype[0]
    probe_angle = genotype[1]
    response_angle = genotype[2]
    distance = genotype[3]
    speed = genotype[4]

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
    num_grid_rows = screen.get_height() // grid_size
    num_grid_cols = screen.get_width() // grid_size
    grid = [[deque() for _ in range(num_grid_cols)] for _ in range(num_grid_rows)]  # Grid initialized with deques





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
    # emitter_x = 900  # Keep the emitter towards the right side
    # emitter_y = 200  # Vertical middle of the screen
    emitter_x = np.random.randint(900, 1100)  # Randomly vary the x-coordinate
    emitter_y = np.random.randint(100, 700)  # Randomly vary the y-coordinate
    emitter_radius = 10  # Radius of the larger sphere

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

    simulation_start_time = pygame.time.get_ticks()  # Get the start time of the simulation
    simulation_time_limit = 60000  # Set a time limit for the simulation (in milliseconds)
    opening_countdown = 0  # Countdown to the opening of the prongs

    olfactory_entity = OlfactoryEntity(100, 400, length, probe_angle, response_angle, distance, speed, network)
    entity_start_time = 2000

    first_draw = True

    # Simulation loop for the current genotype
    while True:
        current_time = pygame.time.get_ticks()
        if current_time - simulation_start_time > simulation_time_limit:
            
            # Punish the entity for not finding the emitter in time
            if olfactory_entity.particle_count != 0:
                punishment = (100 * calculate_distance(emitter_x, emitter_y, olfactory_entity.x, olfactory_entity.y)) / (olfactory_entity.particle_count / 2)
            else:
                punishment = 100 * calculate_distance(emitter_x, emitter_y, olfactory_entity.x, olfactory_entity.y)
            reward_signal -= punishment
            print(f'Punishment for not finding the emitter in time: {punishment}. Ending simulation.')

            break  # Exit the simulation loop after the time limit

        



        # Simulation boundaries
        BOUNDARY_LEFT = 0
        BOUNDARY_RIGHT = 1200
        BOUNDARY_TOP = 0
        BOUNDARY_BOTTOM = 800

        if olfactory_entity.x < BOUNDARY_LEFT or olfactory_entity.x > BOUNDARY_RIGHT or olfactory_entity.y < BOUNDARY_TOP or olfactory_entity.y > BOUNDARY_BOTTOM:
            
            # Punish the entity for going out of bounds
            if olfactory_entity.particle_count != 0:
                punishment = (100 * calculate_distance(emitter_x, emitter_y, olfactory_entity.x, olfactory_entity.y)) / (olfactory_entity.particle_count / 2)
            else:
                punishment = 100 * calculate_distance(emitter_x, emitter_y, olfactory_entity.x, olfactory_entity.y)
            reward_signal -= punishment
            print(f"Punishment for going out of bounds: {punishment}. Ending simulation.")
            return {
                'collided': False,
                'collision_time': None,
                'final_position': (olfactory_entity.x, olfactory_entity.y),
                'simulation_time': pygame.time.get_ticks() - simulation_start_time,  # Total simulation time
                'emitter_position': (emitter_x, emitter_y),
                'emitter_radius': emitter_radius,
                'particle_count': olfactory_entity.particle_count
            }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Simulation logic (particle movement, drawing, etc.) goes here
        # Make sure to update the screen with pygame.display.flip() and limit the frame rate with clock.tick()
    
        # Clear screen
        screen.fill((0, 0, 0))

        # Reset grid for spatial partitioning
        grid = [[deque() for _ in range(num_grid_cols)] for _ in range(num_grid_rows)]  # Reset grid

        # Draw the larger red sphere from which particles are emitted
        pygame.draw.circle(screen, (200, 0, 0), (emitter_x, emitter_y), emitter_radius)

        # Emit white particles
        emit_white_particles(white_particle_rate)  # Emit n white particles per frame

        # Get current time
        current_time = pygame.time.get_ticks()

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
                olfactory_entity.collision_time = pygame.time.get_ticks() - simulation_start_time
                
                # Reward the entity for finding the emitter
                reward_signal += 100
                
                print("Collision with emitter at time:", olfactory_entity.collision_time)
                return {
                    'collided': False,
                    'collision_time': None,
                    'final_position': (olfactory_entity.x, olfactory_entity.y),
                    'simulation_time': pygame.time.get_ticks() - simulation_start_time,  # Total simulation time
                    'emitter_position': (emitter_x, emitter_y),
                    'emitter_radius': emitter_radius,
                    'particle_count': olfactory_entity.particle_count
                }

        opening_countdown += 1

        if opening_countdown > 600:
            if olfactory_entity.timesteps_since_touch > 1200:

                # Punish the entity for getting lost
                if olfactory_entity.particle_count != 0:
                    punishment = (100 * calculate_distance(emitter_x, emitter_y, olfactory_entity.x, olfactory_entity.y)) / (olfactory_entity.particle_count / 2)
                else:
                    punishment = 100 * calculate_distance(emitter_x, emitter_y, olfactory_entity.x, olfactory_entity.y)
                reward_signal -= punishment
                print(f"Punishment for getting lost: {punishment}. Ending simulation.")

                return {
                    'collided': False,
                    'collision_time': None,
                    'final_position': (olfactory_entity.x, olfactory_entity.y),
                    'simulation_time': pygame.time.get_ticks() - simulation_start_time,  # Total simulation time
                    'emitter_position': (emitter_x, emitter_y),
                    'emitter_radius': emitter_radius,
                    'particle_count': olfactory_entity.particle_count
                }
        
        olfactory_entity.timesteps_since_touch += 1
        

        # Inside the main simulation loop
        for particle in list(white_particles):
            if update_particle(particle, grid):
                white_particles.remove(particle)

        # Inside the main simulation loop, where you're iterating over red_particles
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
                pygame.draw.circle(screen, (255, 255, 255), (int(particle[0]), int(particle[1])), particle_radius)
        white_particles = temp_white_particles  # Replace the original deque with the updated one

        # Similarly, update and draw red particles
        temp_red_particles = deque()  # Temporary deque to hold updated particles
        for particle in red_particles:
            if not update_particle(particle, grid):  # Passing 'grid' as an argument
                temp_red_particles.append(particle)
                # Draw the particle after confirming it's still within bounds
                pygame.draw.circle(screen, (255, 0, 0), (int(particle[0]), int(particle[1])), particle_radius)
        red_particles = temp_red_particles  # Replace the original deque with the updated one

        # Update and draw the olfactory tracking entity

        if current_time - simulation_start_time > entity_start_time:
            if first_draw == True:
                olfactory_entity.draw(screen)
                first_draw = False
            olfactory_entity.update(list(red_particles))  # Pass a list of red_particles for sensing
            olfactory_entity.draw(screen)

        if reward_signal != 0:
            Network.modify_learning(network, reward_signal)
            reward_signal = 0

        # Display trial, candidate number, epoch number, architecture, learning rate, and decay rate
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
        clock.tick(120)  # Limit to 60 frames per second

    return {
        'collided': olfactory_entity.collided_with_emitter,
        'collision_time': olfactory_entity.collision_time if olfactory_entity.collided_with_emitter else None,
        'final_position': (olfactory_entity.x, olfactory_entity.y),
        'simulation_time': pygame.time.get_ticks() - simulation_start_time,  # Total simulation time
        'emitter_position': (emitter_x, emitter_y),
        'emitter_radius': emitter_radius,
        'particle_count': olfactory_entity.particle_count
    }

def calculate_distance(emitter_x, emitter_y, final_x, final_y):
    # Calculate the actual distance between the emitter and the final position
    actual_distance = np.hypot(final_x - emitter_x, final_y - emitter_y)
    
    # Calculate the maximum possible distance in the simulation space
    max_distance = np.hypot(1200, 800)
    
    # Normalize the actual distance to a 1-10 scale
    # The scale is inverted since a higher score means farther away
    scaled_distance = 10 - (actual_distance / max_distance * 9)
    
    # Ensure the scaled distance is within the bounds of 1-10
    scaled_distance = max(min(scaled_distance, 10), 1)
    
    return scaled_distance
    
def calculate_fitness(simulation_data, time_limit=60000):
    emitter_x, emitter_y = simulation_data['emitter_position']
    # If the entity collided, fitness is the time until collision
    if simulation_data['collided']:
        simulation_time = simulation_data['collision_time']
        particle_count = simulation_data['particle_count'] / 10
        if particle_count == 0:
            return time_limit
        if particle_count > 0:
            return simulation_time / (particle_count / 2)
    else:
        # If the entity did not collide, calculate the final distance from the emitter
        final_x, final_y = simulation_data['final_position']
        particle_count = simulation_data['particle_count'] / 10
        final_distance = np.hypot(final_x - emitter_x, final_y - emitter_y) * 10
        # Add the time limit to the distance to ensure distance-based fitness
        # always exceeds time-based fitness and place them on a single continuum
        if particle_count == 0:
            return time_limit + final_distance
        if particle_count > 0:
            return (time_limit + final_distance) / (particle_count / 2)


num_epochs = 2  # Set the number of epochs for the evolution process
population_size = 12  # Ensure this is divisible by 2 for simplicity
mutation_rate = 0.1  # Mutation rate for the reproduction process


# Initialize the population
population = [init_population(1)[0] for _ in range(population_size)]



# Function to generate a random name
def generate_random_name(length=6):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

# Function to round genotype values
def round_genotype(genotype):
    return tuple(round(g, 1) for g in genotype[:4])


    



# Initialize a dictionary to store individual fitness along with their genotype and heritage
individual_data = {}

all_trial_data = []  # This will collect all trial data


for epoch in range(num_epochs):
    print(f"EPOCH {epoch+1}/{num_epochs}")

    fitness_scores = []
    for individual in population:
        #print only weights for each individual
        individual_name = generate_random_name() + " " + generate_random_name()
        print(f"Candidate number {population.index(individual) + 1} of {len(population)}: {individual_name}")

        # Initiate network
        weights = individual[5]
        architecture = individual[6]
        depth = individual[7]
        parameters = individual[8]
        recurrence = individual[9]

        network = Network(architecture, depth, parameters, recurrence)


        network.construct(weights, parameters)



        # Run trials and calculate average fitness
        total_fitness = 0
        total_trials = 250
        for _ in range(total_trials):  # Two trials per individual
            network.reset_membranes()
            print(f"Running trial {_ + 1} for {individual_name}")
            simulation_data = run_simulation(individual, screen, clock, _ + 1, total_trials, population.index(individual) + 1, len(population), epoch + 1, num_epochs, network)
            print(f"Particle count: {simulation_data['particle_count']}")
            fitness = calculate_fitness(simulation_data)
            print(f"Fitness: {fitness}")
            trial_data = {
                'trial_number': _ + 1,
                'individual_name': individual_name,
                'particle_count': simulation_data['particle_count'],
                'fitness': fitness,
            }
            all_trial_data.append(trial_data)
            total_fitness += fitness
            # Right after calculating the fitness for a trial
            with open(trial_csv_filename, 'a', newline='') as trial_csvfile:
                trial_writer = csv.DictWriter(trial_csvfile, fieldnames=trial_fieldnames)
                trial_writer.writerow(trial_data)

        avg_fitness = total_fitness / 2
        print(f"Average fitness: {avg_fitness}")


        individual[5] = network.retrieve_weights()


        # Check if individual is new or updating
        if individual_name not in individual_data:
            rounded_genotype = round_genotype(individual)
            individual_data[individual_name] = {
                'epoch': epoch + 1,
                'genotype': rounded_genotype,
                'fitness': avg_fitness,
                'heritage': 'origin',
                'genotype': individual
            }
        else:
            individual_data[individual_name]['fitness'] = avg_fitness  # Update existing fitness
        
        fitness_scores.append((individual_name, individual, avg_fitness))


    # Sort the population based on fitness and select the top 3 individuals
    sorted_population = sorted(fitness_scores, key=lambda x: x[2], reverse=False)
    top_individuals = sorted_population[:3]

    # Reproduce the top individuals to refill the population
    new_population = []

    # Asexual reproduction
    # for parent_name, parent, _ in top_individuals:
    #     child1, child2, child3, child4 = asexual_reproduction(parent, mutation_rate)
    #     for child in [child1, child2, child3, child4]:
    #         child_name = generate_random_name() + " " + generate_random_name()
    #         new_population.append(child)
    #         individual_data[child_name] = {
    #             'genotype': round_genotype(child),
    #             'fitness': None,  # Placeholder, actual fitness to be calculated in the next epoch
    #             'heritage': f'child of {parent_name}',
    #             'epoch': epoch + 1
    #         }

    # Reproduction
    for i in range(len(top_individuals)):
        for j in range(i + 1, len(top_individuals)):  # Start from i + 1 to avoid pairing with itself
            mother = top_individuals[i][1]
            father = top_individuals[j][1]
            children = reproduction(mother, father, 0.1)  # Assume this now returns a list of 4 children
            for child in children:  # Iterate over each child in the list
                child_name = generate_random_name() + " " + generate_random_name()
                new_population.append(child)
                individual_data[child_name] = {
                    'genotype': child,
                    'fitness': None,  # Placeholder, actual fitness to be calculated in the next epoch
                    'heritage': f'child of {top_individuals[i][0]} and {top_individuals[j][0]}',
                    'epoch': epoch + 1
                }

    print(f"Number of offspring: {len(new_population)}")
    population = new_population  # Update the population for the next epoch

# Writing to CSV logic remains unchanged
csv_filename = 'individual_data.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['name', 'genotype', 'fitness', 'heritage', 'epoch']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for individual_name, data in individual_data.items():
        writer.writerow({
            'name': individual_name,
            'fitness': data['fitness'],
            'heritage': data['heritage'],
            'epoch': data['epoch'],
            'genotype': data['genotype']
        })



print(f"Individual data exported to {csv_filename}.")
