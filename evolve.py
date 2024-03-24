import numpy as np
import random
import mlx.core as mx
import copy
from paperwork import Paperwork

def set_seed(seed):
    np.random.seed(seed)

class Individual:
    def __init__(self, architecture, initial_seed, name, recurrent=False):
        self.architecture = architecture
        self.initial_seed = initial_seed
        self.mutation_history = []
        self.name = name
        self.fitness = []
        self.novelty_score = None

        self.avg_fitness = 0
        self.recurrent = recurrent

    def initialise_weights(self):
        set_seed(self.initial_seed)
        weights = []
        recurrent_weights = []
        for i in range(len(self.architecture) - 1):
            layer_weights = mx.array(np.random.uniform(-1, 1, [self.architecture[i], self.architecture[i + 1]]))
            weights.append(layer_weights)
            if self.recurrent and i < len(self.architecture) - 2:
                layer_recurrent_weights = mx.array(np.random.uniform(-1, 1, [self.architecture[i + 1], self.architecture[i]]))
                recurrent_weights.append(layer_recurrent_weights)
        return weights, recurrent_weights

    def rehydrate(self):
        weights, recurrent_weights = self.initialise_weights()
        if len(self.mutation_history) > 0:
            for generation in range(len(self.mutation_history)):
                mutation_seed, mutation_strength = self.mutation_history[generation]
                weights, recurrent_weights = Evolution.mutation(weights, recurrent_weights, mutation_seed, mutation_strength)
            return weights, recurrent_weights
        else:
            return weights, recurrent_weights

class Population:
    def __init__(self, size, architecture, recurrent, individuals=None):
        if individuals is None:
            self.individuals = []
            for i in range(size):
                name = Paperwork.generate_random_name()
                self.individuals.append(Individual(architecture, np.random.randint(1e6), name, recurrent))
        else:
            self.individuals = individuals

class Evolution:
    def reproduction(individual, mutation_strength):
        mutation_seed = random.randint(0, int(1e6))
        new_individual = copy.deepcopy(individual)
        new_individual.name = Paperwork.generate_random_name()
        new_individual.mutation_history = individual.mutation_history.copy()  # Copy the entire mutation history
        new_individual.mutation_history.append([mutation_seed, mutation_strength])  # Append the new mutation
        return new_individual

    def mutation(weights, recurrent_weights, mutation_seed, mutation_strength):
        set_seed(mutation_seed)
        mutated_weights = []
        mutated_recurrent_weights = []
        for layer in weights:
            layer = mx.array(layer)
            mutation = mx.array(np.random.randn(*layer.shape) * mutation_strength)
            mutated_matrix = mx.add(layer, mutation)
            mutated_weights.append(mutated_matrix)
        for layer in recurrent_weights:
            layer = mx.array(layer)
            mutation = mx.array(np.random.randn(*layer.shape) * mutation_strength)
            mutated_matrix = mx.add(layer, mutation)
            mutated_recurrent_weights.append(mutated_matrix)
        return mutated_weights, mutated_recurrent_weights