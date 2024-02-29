import sys
import numpy as np
import random
import string
import csv
import time
import os
from tqdm import tqdm
import pygame
import pandas as pd
import ast
import re
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
        fieldnames = ['epoch', 'name', 'fitness', 'architecture', 'recurrent_layer', 'recurrence_type', 'learning_rate', 'eligibility_decay', 'heritage', 'weights']
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
                'recurrence_type': individual[10],
                'learning_rate': individual[8][0], 
                'eligibility_decay': individual[8][1],
                'recurrent_layer': individual[9],
                'heritage': individual[11],
                'weights': individual[5]
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
def retrieve_population(source, selection):
    df = pd.read_csv(source)

    last_epoch = df['epoch'].max()
    df_last_epoch = df[df['epoch'] == last_epoch]

    df_sorted = df_last_epoch.sort_values(by='fitness', ascending=True)
    
    top_n = df_sorted.head(selection)
    
    results = top_n[['name', 'weights', 'recurrent_layer', 'recurrence_type', 'learning_rate', 'eligibility_decay']]

    output = []

    length = 120
    probe_angle = 55 * (np.pi / 180)
    response_angle = 3 * (np.pi / 180)
    distance = 3
    speed = 3

    for index, row in results.iterrows():
        architecture, depth = interpret_weights(row['weights'])
        weights = ast.literal_eval(row['weights'])

        new_output = [length, probe_angle, response_angle, distance, speed]
        # print(df.columns)
        parameters = [row['learning_rate'], row['eligibility_decay']]
        name = row['name']
        print(f"Name: {name}")
        new_output.extend([weights, architecture, depth, parameters, row['recurrent_layer'], row['recurrence_type'], name])
        output.append(new_output)
    
    return output

def interpret_weights(weights_str):
    weights = ast.literal_eval(weights_str)
    
    neurons_per_layer = {}

    for key in weights.keys():
        if '_rec' in key:
            layer = int(re.search(r'l(\d+)', key).group(1))
            neurons_per_layer.setdefault(layer, 0)
        else:
            pre, post = key.split('_')[0:2]
            layer_pre = int(re.search(r'l(\d+)', pre).group(1))
            neuron_pre = int(re.search(r'n(\d+)', pre).group(1))
            layer_post = int(re.search(r'l(\d+)', post).group(1))
            neuron_post = int(re.search(r'n(\d+)', post).group(1))
            
            neurons_per_layer.setdefault(layer_pre, 0)
            neurons_per_layer[layer_pre] = max(neurons_per_layer[layer_pre], neuron_pre + 1)
            neurons_per_layer.setdefault(layer_post, 0)
            neurons_per_layer[layer_post] = max(neurons_per_layer[layer_post], neuron_post + 1)
    
    total_layers = len(neurons_per_layer)
    architecture = [neurons_per_layer[layer] for layer in sorted(neurons_per_layer)]
    
    return architecture, total_layers

def simulate_individual(individual, population_index, population_size, epoch, num_epochs, num_trials, headless, screen, clock, time, mode):
    if mode == 'new' and epoch == 0:
        individual_name = generate_random_name()
        individual.append(individual_name)
        # print(f"Candidate number {population_index + 1} of {population_size}: {individual_name}")
    elif epoch > 0:
        individual_name = generate_random_name()
        individual[11] = individual[11] + " " + individual_name
        # print(f"Epoch {epoch + 1}, candidate number {population_index + 1} of {population_size}")
    elif mode == 'continue':
        individual_name = individual[11]
        # print(f"Epoch {epoch + 1}, candidate number {population_index + 1} of {population_size}")

    # Initiate network
    weights, architecture, depth, parameters, recurrence, recurrence_type = individual[5:11]
    # print(weights)
    # print(f"Architecture: {architecture}, Depth: {depth}, Parameters: {parameters}, Recurrence: {recurrence}")
    network = Network(architecture, depth, parameters, recurrence, recurrence_type)
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

def parallel_simulations(population, epoch, num_epochs, num_trials, num_processes, screen, clock, time, mode):
    # Use ProcessPoolExecutor to run simulations in parallel
    try:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for index, individual in enumerate(population):
                headless = True
                futures.append(executor.submit(simulate_individual, individual, index, len(population), epoch, num_epochs, num_trials, headless, screen, clock, time, mode))
            
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

def evolutionary_system(population, selection, progeny, epoch, num_epochs, num_trials, processes, headless, screen, clock, time, mode):
    if mode == 'new' or epoch > 0:
        if headless:
            epoch_data = parallel_simulations(population, epoch, num_epochs, num_trials, processes, screen, clock, time, mode)

        if not headless:
            for i, individual in enumerate(population):
                epoch_data = simulate_individual(individual, i, len(population), epoch, num_epochs, num_trials, headless, screen, clock, time, mode)

        epoch_csv(epoch_data)

        survivors = artificial_selection(epoch_data, selection)
    
    elif mode == 'continue':
        survivors = population

    new_population = breeding_program(survivors, progeny, epoch, num_epochs)

    return new_population

def main():

    # Whether to initialise a new population ('new') or continue from a previous run ('continue')
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
    num_epochs = 25

    # Set number of trials for each individual within a generation 
    num_trials = 3

    # Set initial population size
    population_size = 1000

    # Set subsequent population dynamics
    selection = 10
    progeny = 10

    # Setup
    if mode == 'new':
        population = [init_population(1)[0] for _ in range(population_size)]
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
        population = evolutionary_system(population, selection, progeny, epoch, num_epochs, num_trials, processes, headless, screen, clock, time, mode)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()