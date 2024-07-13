import multiprocessing
from methods.evolutionary.neat.neat_handler import NEATHandler
from simulation.simulation import Simulation
from functools import partial
import csv
import os
from datetime import datetime
from ulid import ULID
import pygame
import numpy as np

class Controller:
    def __init__(self, network_type, network_params, method_type, method_params, sim_params, environment):
        print(f'Initialising evolutionary controller with network type {network_type}')
        self.network_type = network_type
        self.network_params = network_params
        self.method_type = method_type
        self.method_params = method_params
        self.sim_params = sim_params

        # Set the method and network parameters
        self.population_size = method_params['population_size']
        print(f'Initialising evolutionary controller with population size {self.population_size}')
        self.n_generations = method_params['n_generations']
        self.n_trials = method_params['n_trials']
        print(f'Running for {self.n_generations} generations with {self.n_trials} trials each')

        # Set the simulation parameters
        self.processes = sim_params['processes']
        self.visualize_best = sim_params['visualize_best']
        self.n_best_genomes = sim_params['n_best_genomes']
        
        # Determine if we're in headless mode
        self.headless = self.processes > 1 or sim_params.get('headless', False)
        if self.processes > 1 and not self.visualize_best:
            print(f'Running with {self.processes} processes')
            self.environment = None
        elif not self.headless and self.visualize_best:
            self.environment = environment
            self.end_env = environment
        elif self.visualize_best:
            self.end_env = environment
            self.environment = None
        else:
            self.environment = environment

        print(f'Running in {"headless" if self.headless else "display"} mode')

        # Initialize the method handler
        self.method_handler = self._init_evolution()
        self.fitness = self.method_handler.fitness

        # Initialize CSV file for logging with unique name
        self.unique_filename = self._generate_unique_filename()
        self.csv_filename = "./records/" + self.unique_filename
        self._initialize_csv()

        # Initialize variables to track overall best genome and fitness
        self.overall_best_genome = None
        self.overall_best_fitness = float('inf')  # Initialize to positive infinity for minimization

        # Run the evolutionary process
        self.run_evolution()

    def _generate_unique_filename(self):
        now = datetime.now()
        date_time = now.strftime("%y-%m-%d-%H-%M")
        unique_id = str(ULID())
        return f"{date_time}_evol-log_{unique_id}.csv"

    def _init_evolution(self):
        if self.method_type == 'NEAT':
            return NEATHandler(self.network_type, self.network_params, self.method_type, self.method_params)
        else:
            raise ValueError(f'Method type {self.method_type} not supported')

    def _initialize_csv(self):
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Generation', 'Genome ID', 'Species ID', 'Average Fitness', 'Network Architecture', 'Best Fitness This Generation', 'Overall Best Fitness'])

    def _log_to_csv(self, generation, genome_id, species_id, avg_fitness, network_description, best_fitness_generation, best_fitness_overall):
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([generation, genome_id, species_id, avg_fitness, network_description, best_fitness_generation, best_fitness_overall])

    def run_trials(self, genome):
        network_architecture, network_description = self.method_handler.convert(genome)
        total_fitness = 0
        trial_number = 0

        for trial in range(self.n_trials):
            network = self.method_handler.initialise_network(network_architecture)
            simulation = Simulation(self.method_handler, network, trial_number)
            results = simulation.run(headless=self.headless, environment=self.environment)
            
            # Calculate fitness based on the results
            if results['collided']:
                fitness = (results['collision_time'] + 1 ** 2)
            else:
                fitness = self.method_handler.calculate_fitness(results)
                fitness = fitness - results['total_concentration']
                if results['total_concentration'] == 0:
                    fitness += 1000
                fitness = fitness + results['simulation_time'] ** 2 * 10
                if fitness == 0:
                    fitness = results['simulation_time'] ** 2
                # If out of bounds, penalize
                if results['out_of_bounds'] or results['is_stalled']:
                    fitness += 500
                fitness = np.clip(fitness, 0, 2000)

            total_fitness += fitness

        average_fitness = (total_fitness / self.n_trials)

        return genome.genome_id, genome.species_id, average_fitness, network_description

    def run_evolution(self):
        for generation in range(self.n_generations):
            print(f'Generation {generation}')
            population = self.method_handler.get_population()
            
            # Run trials using multiprocessing
            if self.processes > 1:
                with multiprocessing.Pool(processes=self.processes) as pool:
                    fitness_results = pool.map(self.run_trials, population)
            else:
                fitness_results = [self.run_trials(genome) for genome in population]
            
            # Process fitness results
            fitness_dict = {}
            for genome_id, species_id, avg_fitness, network_description in fitness_results:
                fitness_dict[genome_id] = avg_fitness
            
            # Evolve the population
            best_genome_gen, best_fitness_gen, best_genome_overall, best_fitness_overall = self.method_handler.evolve(fitness_dict)
            
            # Update overall best if necessary
            if best_fitness_overall < self.overall_best_fitness:
                self.overall_best_fitness = best_fitness_overall
                self.overall_best_genome = best_genome_overall

            print(f'Best fitness for generation {self.method_handler.generation}: {best_fitness_gen}')
            print(f'Overall best fitness: {self.overall_best_fitness}')
            
            # Log results
            for genome_id, species_id, avg_fitness, network_description in fitness_results:
                self._log_to_csv(generation, genome_id, species_id, avg_fitness, network_description, best_fitness_gen, self.overall_best_fitness)

            # Visualize best genomes (if enabled) outside of multiprocessing
            if self.visualize_best:
                best_genomes = sorted(population, key=lambda g: fitness_dict[g.genome_id])[:self.n_best_genomes]
                self.visualize_best_genomes(generation, best_genomes)
            
            self.method_handler.print_best(best_genome_gen)
        
    def visualize_best_genomes(self, generation, best_genomes):
        print(f"Visualizing top {len(best_genomes)} genomes for generation {generation}")
        
        pygame_initialized = False
        screen = None

        try:
            # Initialize Pygame if necessary
            if not pygame.get_init():
                pygame.init()
                pygame_initialized = True
            
            # Create a new environment for visualization
            screen = pygame.display.set_mode((self.sim_params['screen_size']))
            clock = pygame.time.Clock()

            n_trials = self.n_trials
            
            for i, genome in enumerate(best_genomes):
                print(f'Trial fitness for this fella was {best_genomes[i].fitness}')

                for trial_number in range(n_trials):
                    print(f'Trial {trial_number+1} of {n_trials}')
                    network_architecture, _ = self.method_handler.convert(genome)
                    network = self.method_handler.initialise_network(network_architecture)
                    simulation = Simulation(self.method_handler, network, trial_number)
                    
                    # Run the simulation with visualization
                    while simulation.current_time < simulation.config.max_time:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                raise KeyboardInterrupt("Visualization interrupted by user")
                        
                        screen.fill((0, 0, 0))
                        simulation.draw_puffs(screen)
                        simulation.entity.draw(screen)
                        pygame.display.flip()

                        simulation.update()

                        # Check for termination conditions
                        if simulation.out_of_bounds_counter >= simulation.max_out_of_bounds:
                            screen.fill((0, 0, 0))
                            pygame.display.flip()
                            print(f"Genome {i+1} terminated: Out of bounds")
                            break
                        if simulation.stalled_counter > simulation.config.stalled_time:
                            screen.fill((0, 0, 0))
                            pygame.display.flip()
                            print(f"Genome {i+1} terminated: Stalled")
                            break
                        if simulation.collision:
                            print(f"Genome {i+1} terminated: Collision detected")
                            screen.fill((0, 0, 0))
                            pygame.display.flip()
                            break
                        if simulation.current_time >= simulation.config.max_time:
                            screen.fill((0, 0, 0))
                            pygame.display.flip()
                            print(f"Genome {i+1} terminated: Maximum time reached")
                            break

                        clock.tick(60)  # Limit to 60 FPS for visualization

                    print(f"Finished visualizing genome {i+1} of {len(best_genomes)}")
                    trial_number += 1
                    pygame.time.wait(1000)  # Wait for 1 second between genomes

        except Exception as e:
            print(f"An error occurred during visualization: {str(e)}")
        finally:
            # Clean up Pygame resources
            if screen:
                pygame.display.quit()
            if pygame_initialized:
                pygame.quit()
            print("Pygame resources cleaned up")

    def run_simulation(self):
        simulation = Simulation(self.method_handler)
        simulation.run(headless=self.headless, environment=self.environment)
        return simulation.results