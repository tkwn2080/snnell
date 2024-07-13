import numpy as np

from methods.evolutionary.neat.evolution import Evolution
from methods.evolutionary.neat.conversion import GenomeConverter
from methods.evolutionary.neat.network import SNNSimulator

class NEATHandler:
    def __init__(self, network_type, network_params, method_type, method_params):
        print(f'Initialising NEAT handler with network parameters {network_params}')
        self.method = 'evolutionary'

        # Network parameters: whether spiking, number of inputs, outputs, etc.
        self.n_inputs = network_params['n_in']
        self.n_outputs = network_params['n_out']
        self.input_duration = network_params['input_duration']
        self.input_current = network_params['input_current']
        self.propagation_steps = network_params['propagation_steps']

        self.input_parameters = self.input_duration, self.input_current, self.propagation_steps

        # Method parameters: number of generations, number of candidates, etc.
        self.population_size = method_params['population_size']
        self.n_generations = method_params['n_generations']
        self.NEAT_parameters = method_params['NEAT_parameters']

        # Initialise the population
        self.evolution = Evolution(self.population_size, self.n_inputs, self.n_outputs, self.NEAT_parameters)
        self.population = self.evolution.population
        self.fitness = []
        self.generation = 0

        # Initialise the entity
        self.action_type = network_params['action_type']

        # Initialise the environment
        self.environment = None

    def get_population(self):
        return self.evolution.get_population()

    def evolve(self, fitness):
        self.population, best_genome_generation, best_fitness_generation = self.evolution.evolve_one(self.population, fitness)
        self.generation += 1
        self.fitness.clear()
        
        # Get overall best genome and fitness
        best_genome_overall = self.evolution.get_best_genome_overall()
        best_fitness_overall = self.evolution.get_best_fitness_overall()
        
        return best_genome_generation, best_fitness_generation, best_genome_overall, best_fitness_overall

    def convert(self, genome):
        network = GenomeConverter(genome)
        description = network.get_description()
        return network, description

    def input_encoding(self, state):
        current = state * 20 
        return current

    def output_decoding(self, output):
        action = np.sum(output, axis=0)
        return action

    def initialise_network(self, network):
        return SNNSimulator(network.node_array, network.weight_matrix, network.null_mask)

    def get_action(self, network, input_current):
        output = network.propagate(input_current, self.input_duration, self.propagation_steps)
        action = self.output_decoding(output)
        return action

    def update_environment(self, environment):
        self.environment = environment

    def calculate_fitness(self, results):
        final_pos = np.array(results['final_position'])
        emitter_pos = np.array(results['emitter_position'])
        distance = np.linalg.norm(final_pos - emitter_pos)
        if results['collided']:
            return 0  # Minimum distance (best fitness) when collision occurs
        else:
            return distance  # Return the Euclidean distance

    def print_best(self, best_genome):
        best_network = GenomeConverter(best_genome)
        GenomeConverter.print_network_structure(best_network)

    def get_best_genome_overall(self):
        return self.evolution.get_best_genome_overall()

    def get_best_fitness_overall(self):
        return self.evolution.get_best_fitness_overall()

    def get_best_genome_generation(self):
        return self.evolution.get_best_genome_generation()

    def get_best_fitness_generation(self):
        return self.evolution.get_best_fitness_generation()

    def get_current_generation(self):
        return self.evolution.get_current_generation()