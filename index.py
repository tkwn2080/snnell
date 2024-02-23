import sys
import numpy as np
from collections import deque
import random
import string
import csv
import time
import os
from tqdm import tqdm
import pygame
# os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from evolve import init_population, neural_reproduction
from snn import Network
from simulation import run_simulation

import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

# PAPERWORK
def collate_genotype(genotype):
    # genotype_str = f"Learning Rate: {genotype[8][0]}, Eligibility Decay: {genotype[8][1]}, Recurrent Layer: {genotype[9]}"
    weights_str = genotype[5]
    return weights_str

def epoch_csv(epoch_data):
    # Determine if the file needs a header by checking its existence or size
    # epoch_csv_filename = './records/' + time.strftime("%d%m%Y%H") + '_epoch_data.csv'
    epoch_csv_filename = './records/epoch_data.csv'
    write_header = not os.path.exists(epoch_csv_filename) or os.path.getsize(epoch_csv_filename) == 0
    with open(epoch_csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'name', 'fitness', 'architecture', 'recurrent_layer', 'learning_rate', 'eligibility_decay', 'heritage', 'weights']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()

        for data in epoch_data:
            individual = data['individual']
            writer.writerow({
                'epoch': data['epoch'] + 1,
                'name': data['name'],
                'fitness': data['avg_fitness'],
                'architecture': individual[6],
                'recurrent_layer': individual[9],
                'learning_rate': individual[8][0],  # Assuming this is the structure; adjust if necessary
                'eligibility_decay': individual[8][1],
                'recurrent_layer': individual[9],
                'heritage': individual[10],
                'weights': individual[5]
            })
    print(f"Appended data to {epoch_csv_filename}.")

def trial_csv(trial_data):
    # trial_csv_filename = './records/' + time.strftime("%d%m%Y%H") + '_trial_data.csv'
    trial_csv_filename = './records/trial_data.csv'
    trial_fieldnames = ['epoch_number', 'trial_number', 'individual_name', 'particle_count', 'fitness']
    write_header = not os.path.exists(trial_csv_filename) or os.path.getsize(trial_csv_filename) == 0
    with open(trial_csv_filename, 'a', newline='') as trial_csvfile:
        trial_writer = csv.DictWriter(trial_csvfile, fieldnames=trial_fieldnames)
        if write_header:
            trial_writer.writeheader()
        trial_writer.writerow(trial_data)
    # print(f"Appended data to {trial_csv_filename}.")

def calculate_fitness(simulation_data, time_limit=30000):
    emitter_x, emitter_y = simulation_data['emitter_position']
    if simulation_data['collided']:
        simulation_time = simulation_data['collision_time']
        particle_count = simulation_data['particle_count'] / 10
        return simulation_time / particle_count
    else:
        final_x, final_y = simulation_data['final_position']
        final_distance = np.hypot(final_x - emitter_x, final_y - emitter_y)
        return final_distance * 10

    # # If the entity collided, fitness is the time until collision
    # if simulation_data['collided']:
    #     simulation_time = simulation_data['collision_time']
    #     particle_count = simulation_data['particle_count'] / 10
    #     if particle_count == 0:
    #         return time_limit
    #     if particle_count > 0:
    #         return simulation_time / particle_count
    #     # return 0
    # else:
    #     # If the entity did not collide, calculate the final distance from the emitter
    #     final_x, final_y = simulation_data['final_position']
    #     particle_count = simulation_data['particle_count'] / 10
    #     final_distance = np.hypot(final_x - emitter_x, final_y - emitter_y) * 100
    #     # Add the time limit to the distance to ensure distance-based fitness
    #     # always exceeds time-based fitness and place them on a single continuum
    #     if particle_count == 0:
    #         return time_limit + (final_distance * 10)
    #     if particle_count > 0:
    #         return (time_limit + (final_distance * 10)) / (particle_count / 10)
        


        
def generate_random_name(length=6):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


# SIMULATION

def simulate_individual(individual, population_index, population_size, epoch, num_epochs, num_trials, headless, screen, clock, time):
    if epoch == 0:
        individual_name = generate_random_name()
        individual.append(individual_name)
        # print(f"Candidate number {population_index + 1} of {population_size}: {individual_name}")
    elif epoch > 0:
        individual_name = generate_random_name()
        individual[10] = individual[10] + " " + individual_name
        # print(f"Epoch {epoch + 1}, candidate number {population_index + 1} of {population_size}")

    # Initiate network
    weights, architecture, depth, parameters, recurrence = individual[5:10]
    network = Network(architecture, depth, parameters, recurrence)
    network.construct(weights, parameters)

    # Run trials and calculate average fitness
    total_fitness = 0
    total_trials = num_trials
    all_trial_data = []
    for trial in range(total_trials):
        # print(f"Running trial {trial + 1} for {individual_name}")
        simulation_data = run_simulation(individual, trial + 1, total_trials, population_index + 1, population_size, epoch, num_epochs, network, individual_name, headless, screen, clock, time)
        # print(f"Particle count: {simulation_data['particle_count']}")
        fitness = calculate_fitness(simulation_data)
        # print(f"Fitness: {fitness}")
        trial_data = {
            'epoch_number': epoch + 1,
            'trial_number': trial + 1,
            'individual_name': individual_name,
            'particle_count': simulation_data['particle_count'],
            'fitness': fitness,
        }
        trial_csv(trial_data)
        # print(f"Trial {trial + 1} fitness: {fitness}")
        all_trial_data.append(trial_data)
        total_fitness += fitness

    # Save trial data to CSV
    # trial_csv(trial_data)

    avg_fitness = total_fitness / total_trials
    # print(f"Average fitness: {avg_fitness}")

    # Retrieve new weights based on learning
    updated_weights = network.retrieve_weights()
    return population_index, individual_name, updated_weights, avg_fitness, individual[10]

def parallel_simulations(population, epoch, num_epochs, num_trials, num_processes, screen, clock, time):
    # Use ProcessPoolExecutor to run simulations in parallel
    try:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for index, individual in enumerate(population):
                headless = True
                futures.append(executor.submit(simulate_individual, individual, index, len(population), epoch, num_epochs, num_trials, headless, screen, clock, time))
            
            epoch_data = []
            for future in concurrent.futures.as_completed(futures):
                index, individual_name, updated_weights, avg_fitness, heritage = future.result()
                population[index][5] = updated_weights
                population[index].append(heritage)
                epoch_data.append({'epoch': epoch, 'name': individual_name, 'avg_fitness': avg_fitness, 'individual': population[index]})

            return epoch_data
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        sys.exit()
    
def artificial_selection(epoch_data, sample_size):
    # Sort the candidates by average fitness
    sorted_population = sorted(epoch_data, key=lambda x: x['avg_fitness'], reverse=False)
    
    # Return the top n samples
    top_individuals = sorted_population[:sample_size]

    # Restrict to genotype
    survivors = [individual['individual'] for individual in top_individuals]

    return survivors

def breeding_program(population, reproduction_rate, epoch, num_epochs):
    new_population = []
    for individual in population:
        progeny = neural_reproduction(individual, reproduction_rate, epoch, num_epochs)
        new_population.extend(progeny)
    return new_population

def evolutionary_system(population, selection, progeny, epoch, num_epochs, num_trials, processes, headless, screen, clock, time):

    if headless:
        epoch_data = parallel_simulations(population, epoch, num_epochs, num_trials, processes, screen, clock, time)

    if not headless:
        for i, individual in enumerate(population):
            epoch_data = simulate_individual(individual, i, len(population), epoch, num_epochs, num_trials, headless, screen, clock, time)

    epoch_csv(epoch_data)

    survivors = artificial_selection(epoch_data, selection)

    new_population = breeding_program(survivors, progeny, epoch, num_epochs)

    return new_population

def main():

    headless = True
    processes = 7

    num_epochs = 20
    num_trials = 2

    population_size = 1000
    selection = 20
    progeny = 20

    population = [init_population(1)[0] for _ in range(population_size)]

    if headless:
        screen = None
        clock = None
        time = None
    elif not headless:
        pygame.init()
        screen = pygame.display.set_mode((1200, 800), pygame.DOUBLEBUF | pygame.HWSURFACE)
        clock = pygame.time.Clock()
        time = pygame.time


    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        print(f"EPOCH {epoch+1}/{num_epochs}")
        if epoch == 0:
            selection = 20
        else:
            selection = 10
        population = evolutionary_system(population, selection, progeny, epoch, num_epochs, num_trials, processes, headless, screen, clock, time)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()