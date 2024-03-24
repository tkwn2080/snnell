import sys
import numpy as np
import time
from tqdm import tqdm
import pygame
import cProfile

from evolve import Individual, Population, Evolution
from snn import Network
from simulation import Simulation
from paperwork import Paperwork
from selection import Selection

import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

# SIMULATION
def simulate_individual(individual, population_index, population_size, epoch, num_epochs, num_trials, headless, mode, recurrent, environment=None):


    # Run trials and calculate average fitness
    total_fitness = 0
    total_trials = num_trials
    all_trial_data = []
    for trial in range(total_trials):
        emitter_x = np.random.randint(900, 1100)

        if trial % 2:
            emitter_y = np.random.randint(-200, -150)
        else:
            emitter_y = np.random.randint(150, 200)

        neuron_type = 'izhikevich'
        simulation = Simulation(emitter_x, emitter_y, neuron_type, recurrent)
        simulation_data = simulation.simulate('constant', emitter_x, emitter_y, individual, neuron_type, headless, environment, recurrent)
        
        fitness = Paperwork.calculate_fitness(simulation_data)
        individual.fitness.append(fitness)

        trial_data = {
            'epoch_number': epoch + 1,
            'trial_number': trial + 1,
            'individual_name': individual.name,
            'fitness_score': fitness,
            'final_x': simulation_data['final_position'][0],
            'final_y': simulation_data['final_position'][1],
            'emitter_x': simulation_data['emitter_position'][0],
            'emitter_y': simulation_data['emitter_position'][1],
            'behaviour_record': simulation_data['behaviour_record'],
        }

        Paperwork.trial_csv(trial_data)

        all_trial_data.append(trial_data)
        total_fitness += fitness

    avg_fitness = total_fitness / total_trials
    individual.avg_fitness = avg_fitness

    epoch_data = {'epoch': epoch, 'name': individual.name, 'seed': individual.initial_seed, 'avg_fitness': individual.avg_fitness, 'heritage': individual.mutation_history}
    return epoch_data

def parallel_simulations(population, epoch, num_epochs, num_trials, num_processes, mode, recurrent):
    # Use ProcessPoolExecutor to run simulations in parallel
    try:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for index, individual in enumerate(population.individuals):
                headless = True
                futures.append(executor.submit(simulate_individual, individual, index, len(population.individuals), epoch, num_epochs, num_trials, headless, mode, recurrent))
            
            epoch_data = []
            for future in concurrent.futures.as_completed(futures):
                output = future.result()
                # print(output)
                epoch_data.append(output)
            return epoch_data
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        sys.exit()

def breeding_program(population, reproduction_rate, epoch, num_epochs, recurrent):
    mutation_strength = 0.1 - (0.009 * (epoch / num_epochs))

    new_population = Population(0, population.individuals[0].architecture, recurrent)
    for individual in population.individuals:
        for _ in range(reproduction_rate):
            progeny = Evolution.reproduction(individual, mutation_strength)
            new_population.individuals.append(progeny)  # Use append since progeny is a single Individual object
    return new_population

def evolutionary_system(environment,population, selection, selector, progeny, epoch, num_epochs, num_trials, processes, headless, mode, architecture, recurrent):
    if mode == 'new' or epoch > 0:
        epoch_data = []
        if headless:
            output_data = parallel_simulations(population, epoch, num_epochs, num_trials, processes, mode, recurrent)
            epoch_data.extend(output_data)

        if not headless:
            for i, individual in enumerate(population.individuals):
                output_data = simulate_individual(individual, i, len(population.individuals), epoch, num_epochs, num_trials, headless, mode, environment, recurrent)
                epoch_data.append(output_data)

        Paperwork.epoch_csv(epoch_data, epoch, architecture)

        selected = selector.select(population, selection)

        survivors = Population(selection, population.individuals[0].architecture, selected)

    elif mode == 'continue':
        survivors = population

    new_population = breeding_program(survivors, progeny, epoch, num_epochs, recurrent)

    return new_population

def main():

    # Whether to initialise a new population ('new') or continue from a previous run ('continue')
    # There is some sort of problem with continue, where fitness degrades significantly
    mode = 'new'

    # Set headless to True to run without visualisation
    headless = True

    # Set number of processes to run in parallel
    processes = 16
    
    # If multiple processes are used, run headless
    if processes > 1:
        headless = True
    else:
        headless = False

    # Set number of generations
    num_epochs = 30

    # Set number of trials for each individual within a generation 
    num_trials = 2

    # Set initial population size
    population_size = 100

    # Set subsequent population dynamics
    selection = 5
    progeny = 10

    # Set architecture: input and output must be 12 and 4 respectively
    architecture = [14,80,80,80,40,4]

    # Set recurrence
    recurrent = True

    # Set selection type
    selection_type = 'novelty'
    additional_trials = False

    # Setup population, simulation, and selection
    population = Population(population_size, architecture, recurrent)
    
    simulation_settings = {
        'wind_config': 'constant',
        'neuron_type': 'izhikevich',
        'headless': headless,
        'recurrent': recurrent
    }

    selector = Selection(selection_type, additional_trials, simulation_settings=simulation_settings)

    # Setup simulation
    if not headless:
        pygame.init()
        screen = pygame.display.set_mode((1200, 800), pygame.DOUBLEBUF | pygame.HWSURFACE)
        clock = pygame.time.Clock()

        environment = screen, clock
    else:
        environment = None

    # Setup paperwork
    Paperwork.move_existing_files()

    # Run
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        print(f"EPOCH {epoch+1}/{num_epochs}")
        if mode == 'new':
            if epoch == 0:
                n_selection = selection * 2
            else:
                n_selection = selection
        population = evolutionary_system(environment, population, n_selection, selector, progeny, epoch, num_epochs, num_trials, processes, headless, mode, architecture, recurrent)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    cProfile.run('main()')
    # main()