import sys
import numpy as np
from collections import deque
import random
import string
import csv
import time
import os

from evolve import init_population, neural_reproduction
from snn import Network
from simulation import run_simulation

import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

# PAPERWORK
trial_csv_filename = './records/' + time.strftime("%d%m%Y%H") + '_trial_data.csv'
with open(trial_csv_filename, 'w', newline='') as trial_csvfile:
    trial_fieldnames = ['epoch_number', 'trial_number', 'individual_name', 'particle_count', 'fitness']
    trial_writer = csv.DictWriter(trial_csvfile, fieldnames=trial_fieldnames)
    trial_writer.writeheader()

def collate_genotype(genotype):
    # genotype_str = f"Learning Rate: {genotype[8][0]}, Eligibility Decay: {genotype[8][1]}, Recurrent Layer: {genotype[9]}"
    weights_str = genotype[5]
    return weights_str

def epoch_csv(epoch_data):
    # Determine if the file needs a header by checking its existence or size
    epoch_csv_filename = './records/' + time.strftime("%d%m%Y%H") + '_epoch_data.csv'
    write_header = not os.path.exists(epoch_csv_filename) or os.path.getsize(epoch_csv_filename) == 0
    with open(epoch_csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'name', 'fitness', 'heritage', 'learning_rate', 'eligibility_decay', 'recurrent_layer', 'weights']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()

        for data in epoch_data:
            individual = data['individual']
            writer.writerow({
                'epoch': data['epoch'] + 1,
                'name': data['name'],
                'fitness': data['avg_fitness'],
                'heritage': 'N/A',  # Assuming there's no heritage info; adjust as needed
                'learning_rate': individual[8][0],  # Assuming this is the structure; adjust if necessary
                'eligibility_decay': individual[8][1],
                'recurrent_layer': individual[9],
                'weights': individual[5],
            })
    print(f"Appended data to {epoch_csv_filename}.")

def trial_csv(trial_data):
    trial_csv_filename = './records/' + time.strftime("%d%m%Y%H") + '_trial_data.csv'
    trial_fieldnames = ['epoch_number', 'trial_number', 'individual_name', 'particle_count', 'fitness']
    with open(trial_csv_filename, 'a', newline='') as trial_csvfile:
        trial_writer = csv.DictWriter(trial_csvfile, fieldnames=trial_fieldnames)
        trial_writer.writerow(trial_data)
    print('Saved trial data to CSV')

def calculate_fitness(simulation_data, time_limit=45000):
    emitter_x, emitter_y = simulation_data['emitter_position']
    # If the entity collided, fitness is the time until collision
    if simulation_data['collided']:
        # simulation_time = simulation_data['collision_time']
        # particle_count = simulation_data['particle_count'] / 10
        # if particle_count == 0:
        #     return time_limit
        # if particle_count > 0:
        #     return simulation_time / (particle_count / 10)
        return 0
    else:
        # If the entity did not collide, calculate the final distance from the emitter
        final_x, final_y = simulation_data['final_position']
        particle_count = simulation_data['particle_count'] / 10
        final_distance = np.hypot(final_x - emitter_x, final_y - emitter_y) * 100
        # Add the time limit to the distance to ensure distance-based fitness
        # always exceeds time-based fitness and place them on a single continuum
        if particle_count == 0:
            return time_limit + final_distance
        if particle_count > 0:
            return (time_limit + final_distance) / (particle_count / 10)
        
def generate_random_name(length=6):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


# SIMULATION

def simulate_individual(individual, population_index, population_size, epoch, num_epochs, num_trials, headless):
    individual_name = generate_random_name() + " " + generate_random_name()
    print(f"Candidate number {population_index + 1} of {population_size}: {individual_name}")

    # Initiate network
    weights, architecture, depth, parameters, recurrence = individual[5:]
    network = Network(architecture, depth, parameters, recurrence)
    network.construct(weights, parameters)

    # Run trials and calculate average fitness
    total_fitness = 0
    total_trials = num_trials
    all_trial_data = []
    for trial in range(total_trials):
        print(f"Running trial {trial + 1} for {individual_name}")
        simulation_data = run_simulation(individual, trial + 1, total_trials, population_index + 1, population_size, epoch, num_epochs, network, individual_name, headless)
        print(f"Particle count: {simulation_data['particle_count']}")
        fitness = calculate_fitness(simulation_data)
        print(f"Fitness: {fitness}")
        trial_data = {
            'epoch_number': epoch + 1,
            'trial_number': trial + 1,
            'individual_name': individual_name,
            'particle_count': simulation_data['particle_count'],
            'fitness': fitness,
        }
        trial_csv(trial_data)
        all_trial_data.append(trial_data)
        total_fitness += fitness

    # Save trial data to CSV
    # trial_csv(trial_data)

    avg_fitness = total_fitness / total_trials
    print(f"Average fitness: {avg_fitness}")

    # Retrieve new weights based on learning
    updated_weights = network.retrieve_weights()
    return population_index, individual_name, updated_weights, avg_fitness

def parallel_simulations(population, epoch, num_epochs, num_trials, num_processes):
    # Use ProcessPoolExecutor to run simulations in parallel
    try:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for index, individual in enumerate(population):
                headless = True
                futures.append(executor.submit(simulate_individual, individual, index, len(population), epoch, num_epochs, num_trials, headless))
            
            epoch_data = []
            for future in concurrent.futures.as_completed(futures):
                index, individual_name, updated_weights, avg_fitness = future.result()
                population[index][5] = updated_weights
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
    top_individuals = [individual['individual'] for individual in top_individuals]

    return top_individuals

def breeding_program(population, reproduction_rate):
    new_population = []
    for individual in population:
        progeny = neural_reproduction(individual, reproduction_rate)
        new_population.append(progeny)
    return new_population


def evolutionary_system(population, selection, progeny, epoch, num_epochs, num_trials, processes, headless):

    if headless:
        epoch_data = parallel_simulations(population, epoch, num_epochs, num_trials, processes)

    if not headless:
        for i, individual in enumerate(population):
            epoch_data = simulate_individual(individual, i, len(population), epoch, num_epochs, num_trials, headless = False)

    epoch_csv(epoch_data)

    survivors = artificial_selection(epoch_data, selection)

    new_population = breeding_program(survivors, progeny)

    return new_population

    


def main():

    headless = True
    processes = 7

    num_epochs = 10
    num_trials = 5

    population_size = 20
    selection = 4
    progeny = 5

    population = [init_population(1)[0] for _ in range(population_size)]

    for epoch in range(num_epochs):
        print(f"EPOCH {epoch+1}/{num_epochs}")
        population = evolutionary_system(population, selection, progeny, epoch, num_epochs, num_trials, processes, headless)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()