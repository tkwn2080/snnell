import argparse
import csv
import pandas as pd
import numpy as np
import pygame
from evolve import Individual
from simulation import Simulation
from paperwork import Paperwork

def load_epoch_data(epoch_data_file):
    data = pd.read_csv(epoch_data_file)
    return data

def select_individuals(data, selection_type, num_individuals):
    if selection_type == 'fitness':
        # Select individuals from the last epoch based on fitness
        last_epoch = data['epoch'].max()
        last_epoch_data = data[data['epoch'] == last_epoch]
        selected_data = last_epoch_data.nsmallest(num_individuals, 'fitness')
    elif selection_type == 'novelty':
        # Select individuals from the entire file based on novelty
        selected_data = data.nsmallest(num_individuals, 'fitness')
    else:
        raise ValueError(f"Invalid selection type: {selection_type}")
    return selected_data

def reconstruct_individual(row, recurrent=False):
    architecture = eval(row['architecture'])
    initial_seed = int(row['seed'])
    heritage = eval(row['heritage'])
    individual = Individual(architecture, initial_seed, row['name'], recurrent)
    individual.mutation_history = heritage
    return individual

def run_trials(individuals, num_trials, environment, recurrent=False):
    headless = False
    for individual in individuals:
        print(f"Running trials for individual: {individual.name}")
        total_fitness = 0
        for trial in range(num_trials):
            emitter_x = np.random.randint(900, 1100)
            if trial % 2:
                emitter_y = np.random.randint(-150, -100)
            else:
                emitter_y = np.random.randint(100, 150)
            neuron_type = 'izhikevich'
            simulation = Simulation(emitter_x, emitter_y, neuron_type, recurrent)
            simulation_data = simulation.simulate('constant', emitter_x, emitter_y, individual, neuron_type, headless, recurrent, environment)
            fitness = Paperwork.calculate_fitness(simulation_data)
            print(f"Fitness for trial {trial + 1}: {fitness}")
            total_fitness += fitness
        avg_fitness = total_fitness / num_trials
        print(f"Average fitness for individual {individual.name}: {avg_fitness}")

def main():
    # Hardcoded settings
    selection_type = 'novelty'
    num_individuals = 20
    num_trials = 2
    recurrent = True

    pygame.init()
    screen = pygame.display.set_mode((1200, 800), pygame.DOUBLEBUF | pygame.HWSURFACE)
    clock = pygame.time.Clock()
    environment = screen, clock

    epoch_data_file = './records/epoch_data.csv'
    data = load_epoch_data(epoch_data_file)
    selected_data = select_individuals(data, selection_type, num_individuals)
    individuals = [reconstruct_individual(row, recurrent) for _, row in selected_data.iterrows()]
    run_trials(individuals, num_trials, environment, recurrent)

if __name__ == '__main__':
    main()