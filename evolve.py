import numpy as np
import random
import mlx.core as mx
from snn import Network
import copy

def set_seed(seed):
    np.random.seed(seed)

class Individual:
    def __init__(self, architecture, initial_seed):
        self.architecture = architecture
        self.initial_seed = initial_seed
        self.weights = self.initialise_weights()
        self.mutation_history = []
        self.name = ''
        self.fitness = []
        self.avg_fitness = 0

    def initialise_weights(self):
        set_seed(self.initial_seed)
        weights = []
        for i in range(len(self.architecture) - 1):
            layer_weights = np.random.uniform(-1, 1, [self.architecture[i], self.architecture[i + 1]]).tolist()
            weights.append(layer_weights)
        return weights

class Population:
    def __init__(self, size, architecture, individuals=None):
        if individuals is None:
            self.individuals = []
            for i in range(size):
                self.individuals.append(Individual(architecture, np.random.randint(1e6)))
        else:
            self.individuals = individuals

    def selection(self, selection):
        sorted_individuals = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
        top_n = sorted_individuals[:selection]
        survivors = Population(selection, self.individuals[0].architecture, top_n)
        return survivors

class Evolution:
    def reproduction(individual, mutation_strength):
        mutation_seed = random.randint(0, int(1e6))
        new_individual = copy.deepcopy(individual)
        new_individual.weights = Evolution.mutation(individual, mutation_seed, mutation_strength)
        new_individual.mutation_history.append([mutation_seed, mutation_strength])
        return new_individual

    def mutation(individual, mutation_seed, mutation_strength):
        set_seed(mutation_seed)
        mutated_weights = []
        for layer in individual.weights:
            # layer = mx.array(layer)
            # mutation = mx.array(np.random.randn(*layer.shape) * mutation_strength)
            # mutated_matrix = mx.add(layer, mutation)
            # mutated_weights.append(mutated_matrix)

            # Temporary list solution
            mutation = (np.random.randn(*np.array(layer).shape) * mutation_strength).tolist()

            mutated_layer = [[layer_val + mutation_val for layer_val, mutation_val in zip(layer_row, mutation_row)]
                            for layer_row, mutation_row in zip(layer, mutation)]
            mutated_weights.append(mutated_layer)
        return mutated_weights