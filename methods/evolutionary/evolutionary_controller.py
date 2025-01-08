import multiprocessing
from methods.evolutionary.neat.neat_handler import NEATHandler
from simulation.simulation import Simulation
from functools import partial
import csv
import json
from datetime import datetime
from ulid import ULID
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
from tqdm import tqdm

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
        self.current_generation = 0
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
        self.best_genomes_filename = "best_genomes.csv"
        self._initialize_best_genomes_csv()

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

    def _initialize_best_genomes_csv(self):
        with open(self.best_genomes_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['Timestamp', 'Name', 'Best Fitness', 'Network Structure', 'Connection Count']
            writer.writerow(headers)

    def get_network_structure(self, genome):
        input_nodes = len([node for node in genome.nodes if node['node_type'] == 'input'])
        output_nodes = len([node for node in genome.nodes if node['node_type'] == 'output'])
        hidden_nodes = len([node for node in genome.nodes if node['node_type'] == 'hidden'])
        return f"[{input_nodes}, {hidden_nodes}, {output_nodes}]"

    def save_best_genome(self, name, genome, fitness):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        network_structure = self.get_network_structure(genome)
        
        # Sort connections based on their order in the genome
        sorted_connections = sorted(genome.connections, key=lambda c: c.innovation_id)
        
        # Create the base row
        row = [timestamp, name, fitness, network_structure, len(sorted_connections)]
        
        # Add connection information to the row as dictionaries
        for conn in sorted_connections:
            connection_dict = {
                'in_node': conn.in_node,
                'out_node': conn.out_node,
                'weight': conn.weight,
                'enabled': int(conn.enabled),
                'innovation_id': conn.innovation_id
            }
            row.append(json.dumps(connection_dict))

        with open(self.best_genomes_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

    def run_trials(self, genome):
        trial_error = False

        try: # If the genome cannot be converted, return a penalty fitness
            network_architecture, network_description = self.method_handler.convert(genome) 
        except Exception as e:
            print(f"Error converting genome {genome.genome_id}: {str(e)}")
            trial_error = True

        total_fitness = 0
        trial_number = 0

        for trial in range(self.n_trials):
            try:
                network = self.method_handler.initialise_network(network_architecture)
                simulation = Simulation(self.method_handler, network, trial_number, self.current_generation)
                results = simulation.run(headless=self.headless, environment=self.environment)

                # Calculate fitness based on the results
                if results['collided']:
                    fitness = (results['collision_time'] + 1 ** 2)
                    # distance = fitness
                else:
                    fitness = self.method_handler.calculate_fitness(results)
                    distance = fitness
                    fitness = fitness - results['total_concentration']
                    if results['total_concentration'] == 0:
                        fitness += 1000
                    if fitness == 0:
                        fitness = results['simulation_time'] ** 2
                    # If out of bounds, penalize
                    if results['out_of_bounds'] or results['is_stalled']:
                        fitness += 1000
                    fitness = np.clip(fitness, 0, 3000)
                    fitness = fitness + results['simulation_time']

                total_fitness += fitness
                trial_number += 1

            except Exception as e:
                print(f"Error in trial {trial} for genome {genome.genome_id}: {str(e)}")
                trial_error = True

        if trial_error:
            average_fitness = 9999
            return genome.genome_id, genome.species_id, average_fitness, 'Error converting genome'
        else:
            average_fitness = (total_fitness / self.n_trials)
            if average_fitness > 3000:
                print(f"Average fitness too high for genome {genome.genome_id}: {average_fitness}")
            return genome.genome_id, genome.species_id, average_fitness, network_description

    def run_evolution(self):
        for generation in range(self.n_generations):
            print(f'\nGeneration {generation}')
            population = self.method_handler.get_population()
            
            # Run trials using multiprocessing with progress bar for evaluation
            if self.processes > 1:
                with multiprocessing.Pool(processes=self.processes) as pool:
                    fitness_results = list(tqdm(pool.imap(self.run_trials, population), 
                                                total=len(population), 
                                                desc=f"Evaluating genomes (Gen {generation})"))
            else:
                fitness_results = list(tqdm(map(self.run_trials, population), 
                                            total=len(population), 
                                            desc=f"Evaluating genomes (Gen {generation})"))
            
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

            if best_fitness_gen < 500:
                self.save_best_genome(f"Gen_{generation}", best_genome_gen, best_fitness_gen)

            # Visualize best genomes (if enabled) outside of multiprocessing
            if self.visualize_best:
                best_genomes = sorted(population, key=lambda g: fitness_dict[g.genome_id])[:self.n_best_genomes]
                self.visualize_best_genomes(generation, best_genomes, fitness_dict)
            
            self.method_handler.print_best(best_genome_gen)
            self.current_generation += 1
        
    def visualize_best_genomes(self, generation, best_genomes, fitness_dict):
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
                trial_fitness = fitness_dict[genome.genome_id]
                print(f'Trial fitness for this fella was {trial_fitness}')

                for trial_number in range(n_trials):
                    print(f'Trial {trial_number+1} of {n_trials}')
                    network_architecture, _ = self.method_handler.convert(genome)
                    network = self.method_handler.initialise_network(network_architecture)
                    simulation = Simulation(self.method_handler, network, trial_number, self.current_generation)
                    
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

    # def run_simulation(self):
    #     simulation = Simulation(self.method_handler)
    #     simulation.run(headless=self.headless, environment=self.environment)
    #     return simulation.results