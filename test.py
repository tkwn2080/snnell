from tqdm import tqdm
import pygame
import pandas as pd
import numpy as np
import ast
import re

from snn import Network
from simulation import run_simulation

# SETUP
import re
import ast

def interpret_weights(weights_str):
    # Convert string representation of dictionary to actual dictionary
    weights = ast.literal_eval(weights_str)
    
    # Initialize structure to keep track of neurons per layer
    neurons_per_layer = {}

    for key in weights.keys():
        # Handle recurrent connections differently
        if '_rec' in key:
            layer = int(re.search(r'l(\d+)', key).group(1))
            # Ensure this layer is accounted for even if it only has recurrent connections
            neurons_per_layer.setdefault(layer, 0)
        else:
            # Split the key based on layer-neuron structure
            pre, post = key.split('_')[0:2]
            layer_pre = int(re.search(r'l(\d+)', pre).group(1))
            neuron_pre = int(re.search(r'n(\d+)', pre).group(1))
            layer_post = int(re.search(r'l(\d+)', post).group(1))
            neuron_post = int(re.search(r'n(\d+)', post).group(1))
            
            # Update the count of neurons in the layers
            neurons_per_layer.setdefault(layer_pre, 0)
            neurons_per_layer[layer_pre] = max(neurons_per_layer[layer_pre], neuron_pre + 1)
            neurons_per_layer.setdefault(layer_post, 0)
            neurons_per_layer[layer_post] = max(neurons_per_layer[layer_post], neuron_post + 1)
    
    # Calculate total layers and neurons per layer
    total_layers = len(neurons_per_layer)
    architecture = [neurons_per_layer[layer] for layer in sorted(neurons_per_layer)]
    
    return architecture, total_layers

def pull_weights(record_path, n=10):
    # Step 1: Read the CSV file
    df = pd.read_csv(record_path)
    
    # Step 2: Filter for the last epoch
    last_epoch = df['epoch'].max()
    df_last_epoch = df[df['epoch'] == last_epoch]
    
    # Step 3: Sort by fitness in descending order
    df_sorted = df_last_epoch.sort_values(by='fitness', ascending=True)
    
    # Step 4: Select top N rows
    top_n = df_sorted.head(n)
    
    # Step 5: Extract weights and architecture
    results = top_n[['name', 'weights', 'recurrent_layer']]

    output = []

    # Pad with settings
    length = 80
    probe_angle = 55 * (np.pi / 180)
    response_angle = 3 * (np.pi / 180)
    distance = 3
    speed = 3

    for index, row in results.iterrows():
        architecture, depth = interpret_weights(row['weights'])
        weights = ast.literal_eval(row['weights'])

        new_output = [length, probe_angle, response_angle, distance, speed]
        new_output.extend([row['name'], weights, row['recurrent_layer'], architecture, depth])
        output.append(new_output)
    
    # Step 6: Return or print the result
    return output

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

def simulate_individual(individual, population_index, population_size, epoch, num_epochs, num_trials, headless, screen, clock, time):

    name, weights, recurrence, architecture, depth = individual[5:10]
    parameters = [0,0]
    network = Network(architecture, depth, parameters, recurrence)
    network.construct(weights, parameters)

    # Run trials and calculate average fitness
    for trial in range(num_trials):
        print(f"Running trial {trial + 1} for {name}")
        simulation_data = run_simulation(individual, trial + 1, num_trials, population_index + 1, population_size, epoch, num_epochs, network, name, headless, screen, clock, time)
        # print(f"Particle count: {simulation_data['particle_count']}")
        fitness = calculate_fitness(simulation_data)
        print(f"Fitness: {fitness}")

def main():
    
    headless = False
    processes = 1

    # Set top n to test
    top_n = 5

    # Set number of trials for each individual within a generation
    num_trials = 2

    pygame.init()
    screen = pygame.display.set_mode((1200, 800), pygame.DOUBLEBUF | pygame.HWSURFACE)
    clock = pygame.time.Clock()
    time = pygame.time


    record_location = 'records/epoch_data.csv'
    population = pull_weights(record_location, top_n)

    # Run
    for i, individual in enumerate(population):
        simulate_individual(individual, 0, top_n, 0, 1, num_trials, False, screen, clock, time)



if __name__ == "__main__":
    main()