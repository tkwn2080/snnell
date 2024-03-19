import numpy as np
import random
import mlx.core as mx
import copy
from paperwork import Paperwork

def set_seed(seed):
    np.random.seed(seed)

class Individual:
    def __init__(self, architecture, initial_seed, name):
        self.architecture = architecture
        self.initial_seed = initial_seed
        self.mutation_history = []
        self.name = name
        self.fitness = []
        self.avg_fitness = 0

    def initialise_weights(self):
        set_seed(self.initial_seed)
        weights = []
        for i in range(len(self.architecture) - 1):
            layer_weights = mx.array(np.random.uniform(-1, 1, [self.architecture[i], self.architecture[i + 1]]))
            weights.append(layer_weights)
        return weights

    def rehydrate(self):
        weights = self.initialise_weights()
        if len(self.mutation_history) > 0:
            for generation in range(len(self.mutation_history)):
                mutation_seed, mutation_strength = self.mutation_history[generation]
                weights = Evolution.mutation(weights, mutation_seed, mutation_strength)
            return weights
        else:
            return weights

class Population:
    def __init__(self, size, architecture, individuals=None):
        if individuals is None:
            self.individuals = []
            for i in range(size):
                name = Paperwork.generate_random_name()
                self.individuals.append(Individual(architecture, np.random.randint(1e6), name))
        else:
            self.individuals = individuals

class Evolution:
    def reproduction(individual, mutation_strength):
        mutation_seed = random.randint(0, int(1e6))
        new_individual = copy.deepcopy(individual)
        new_individual.name = Paperwork.generate_random_name()
        new_individual.mutation_history.append([mutation_seed, mutation_strength])
        return new_individual

    def mutation(weights, mutation_seed, mutation_strength):
        set_seed(mutation_seed)
        mutated_weights = []
        for layer in weights:
            layer = mx.array(layer)
            mutation = mx.array(np.random.randn(*layer.shape) * mutation_strength)
            mutated_matrix = mx.add(layer, mutation)
            mutated_weights.append(mutated_matrix)
        return mutated_weights