import sys
import numpy as np
import random
import string
import csv
import time
import os
from tqdm import tqdm
import pygame
# import pandas as pd
# import ast
# import re
# os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from evolve import Individual, Population, Evolution
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

def epoch_csv(epoch_data, epoch, architecture):
    # Determine if the file needs a header by checking its existence or size
    # epoch_csv_filename = './records/' + time.strftime("%d%m%Y%H") + '_epoch_data.csv'
    epoch_csv_filename = './records/epoch_data.csv'
    write_header = not os.path.exists(epoch_csv_filename) or os.path.getsize(epoch_csv_filename) == 0
    with open(epoch_csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'name', 'seed', 'fitness', 'architecture', 'heritage']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()

        # print(f'Epoch data: {epoch_data}')

        for data in epoch_data:
            writer.writerow({
                'epoch': epoch + 1,
                'name': data['name'],
                'seed': data['seed'],
                'fitness': data['avg_fitness'],
                'architecture': architecture,
                'heritage': data['heritage'],
            })

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
        collision_time = simulation_data['collision_time']
        particle_count = simulation_data['particle_count'] / 10
        return collision_time / particle_count
    else:
        final_x, final_y = simulation_data['final_position']
        simulation_time = simulation_data['simulation_time']
        final_distance = np.hypot(final_x - emitter_x, final_y - emitter_y)
        return simulation_time + final_distance * 10
  
def generate_random_name(length=6):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

# SIMULATION
def simulate_individual(individual, population_index, population_size, epoch, num_epochs, num_trials, headless, screen, clock, time, mode):
    if epoch == 0:
        individual.name = generate_random_name()
    else:
        individual.name = individual.name + " " + generate_random_name()

    # Initiate network
    network = Network(individual)

    # Run trials and calculate average fitness
    total_fitness = 0
    total_trials = num_trials
    all_trial_data = []
    for trial in range(total_trials):
        simulation_data = run_simulation(individual, trial + 1, total_trials, population_index + 1, population_size, epoch, num_epochs, network, headless, screen, clock, time)
        fitness = calculate_fitness(simulation_data)
        individual.fitness.append(fitness)

        trial_data = {
            'epoch_number': epoch + 1,
            'trial_number': trial + 1,
            'individual_name': individual.name,
            'particle_count': simulation_data['particle_count'],
            'fitness': fitness,
        }
        trial_csv(trial_data)

        all_trial_data.append(trial_data)
        total_fitness += fitness


    avg_fitness = total_fitness / total_trials
    individual.avg_fitness = avg_fitness

    epoch_data = {'epoch': epoch, 'name': individual.name, 'seed': individual.initial_seed, 'avg_fitness': individual.avg_fitness, 'heritage': individual.mutation_history}
    # print(epoch_data)
    return epoch_data

def parallel_simulations(population, epoch, num_epochs, num_trials, num_processes, screen, clock, time, mode):
    # Use ProcessPoolExecutor to run simulations in parallel
    try:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for index, individual in enumerate(population.individuals):
                headless = True
                futures.append(executor.submit(simulate_individual, individual, index, len(population.individuals), epoch, num_epochs, num_trials, headless, screen, clock, time, mode))
            
            epoch_data = []
            for future in concurrent.futures.as_completed(futures):
                output = future.result()
                # print(output)
                epoch_data.append(output)
            return epoch_data
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        sys.exit()

def breeding_program(population, reproduction_rate, epoch, num_epochs):
    mutation_strength = 0.1 - (0.009 * (epoch / num_epochs))

    new_population = Population(0, population.individuals[0].architecture)
    for individual in population.individuals:
        for _ in range(reproduction_rate):
            progeny = Evolution.reproduction(individual, mutation_strength)
            new_population.individuals.append(progeny)  # Use append since progeny is a single Individual object
    return new_population

def evolutionary_system(population, selection, progeny, epoch, num_epochs, num_trials, processes, headless, screen, clock, time, mode, architecture):
    if mode == 'new' or epoch > 0:
        epoch_data = []
        if headless:
            print(f"Running in headless mode")
            output_data = parallel_simulations(population, epoch, num_epochs, num_trials, processes, screen, clock, time, mode)
            epoch_data.extend(output_data)
            print(epoch_data)

        if not headless:
            for i, individual in enumerate(population.individuals):
                output_data = simulate_individual(individual, i, len(population.individuals), epoch, num_epochs, num_trials, headless, screen, clock, time, mode)
                epoch_data.append(output_data)
                print(epoch_data)

        epoch_csv(epoch_data, epoch, architecture)

        survivors = population.selection(selection)
    
    elif mode == 'continue':
        survivors = population

    new_population = breeding_program(survivors, progeny, epoch, num_epochs)

    return new_population

def main():

    # Whether to initialise a new population ('new') or continue from a previous run ('continue')
    # There is some sort of problem with continue, where fitness degrades significantly
    mode = 'new'

    # Set headless to True to run without visualisation
    headless = True

    # Set number of processes to run in parallel
    processes = 7
    
    # If multiple processes are used, run headless
    if processes > 1:
        headless = True
    else:
        headless = False

    # Set number of generations
    num_epochs = 20

    # Set number of trials for each individual within a generation 
    num_trials = 5

    # Set initial population size
    population_size = 1000

    # Set subsequent population dynamics
    selection = 10
    progeny = 10

    # Set architecture
    architecture = [6,1000,1000,1000,1000,4]

    # Setup
    if mode == 'new':
        print("Initialising new population")
        population = Population(population_size, architecture)
    elif mode == 'continue':
        source = 'records/epoch_data.csv'
        population = retrieve_population(source, selection)

    if headless:
        screen = None
        clock = None
        time = None
    elif not headless:
        pygame.init()
        screen = pygame.display.set_mode((1200, 800), pygame.DOUBLEBUF | pygame.HWSURFACE)
        clock = pygame.time.Clock()
        time = pygame.time

    # Run
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        print(f"EPOCH {epoch+1}/{num_epochs}")
        if mode == 'new':
            if epoch == 0:
                selection = 20
            else:
                selection = 10
        population = evolutionary_system(population, selection, progeny, epoch, num_epochs, num_trials, processes, headless, screen, clock, time, mode, architecture)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()