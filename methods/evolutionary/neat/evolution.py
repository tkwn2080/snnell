from methods.evolutionary.neat.population import Population, Speciation, Reproduction, Mutation
from methods.evolutionary.neat.genome import Genome, InnovationTracker
import numpy as np
import random

class Evolution:
    def __init__(self, population_size, n_inputs, n_outputs, parameters):
        self.population_size = population_size
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.parameters = parameters
        self.higher_fitness_is_better = False  # Hardcoded to False for distance-based fitness
        
        self.population = Population(population_size, n_inputs, n_outputs, parameters)
        self.speciation = Speciation(parameters['speciation'])
        self.reproduction = Reproduction(parameters['reproduction'])
        self.mutation = Mutation(parameters['mutation'])
        
        self.current_generation = 0
        self.best_fitness_overall = float('inf')  # Initialize to positive infinity for minimization
        self.best_fitness_generation = float('inf')  # Best fitness for the current generation
        self.best_genome_overall = None
        self.best_genome_generation = None

    def is_better_fitness(self, new_fitness, old_fitness):
        return new_fitness <= old_fitness  # Lower fitness is better

    def evolve_one(self, population, fitness):
        # Update best genome for the generation
        fitness_values = list(fitness.values())
        self.best_fitness_generation = min(fitness_values)  # Use min for minimization
        
        # Choose the best genome for this generation
        best_genomes_generation = [genome for genome in self.population.genomes 
                                   if fitness[genome.genome_id] == self.best_fitness_generation]
        self.best_genome_generation = random.choice(best_genomes_generation)
        
        # Update overall best if the generation best is better
        if self.is_better_fitness(self.best_fitness_generation, self.best_fitness_overall):
            self.best_fitness_overall = self.best_fitness_generation
            self.best_genome_overall = Genome(self.best_genome_generation.n_in, 
                                              self.best_genome_generation.n_out, 
                                              self.best_genome_generation.innovation_tracker, 
                                              source=[self.best_genome_generation.connections, 
                                                      self.best_genome_generation.nodes])

        # Speciate
        species = self.speciation.speciate(self.population.genomes)
        print(f"Number of species for generation {self.current_generation}: {len(species)}")
        
        # Reproduce
        new_population = self.reproduction.reproduce(self.population.genomes, species, fitness)
        print(f"New population size at generation {self.current_generation}: {len(new_population)}")
        
        # Mutate
        mutated_population = []
        for genome in new_population:
            mutated_genome = self.mutation.mutate(genome)
            mutated_population.append(mutated_genome)
        
        # Update population
        self.population.replace_population(mutated_population)
        self.current_generation += 1
        
        return self.population, self.best_genome_generation, self.best_fitness_generation

    def evolve(self, fitness_function, n_generations):
        for _ in range(n_generations):
            self.evolve_one_generation(fitness_function)
        return self.best_genome_overall, self.best_fitness_overall

    def get_population(self):
        return self.population.genomes

    def get_best_genome_overall(self):
        return self.best_genome_overall

    def get_best_fitness_overall(self):
        return self.best_fitness_overall

    def get_best_genome_generation(self):
        return self.best_genome_generation

    def get_best_fitness_generation(self):
        return self.best_fitness_generation

    def get_current_generation(self):
        return self.current_generation