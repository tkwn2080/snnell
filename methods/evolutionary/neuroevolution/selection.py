import mlx.core as mx
import pandas as pd
import numpy as np
import random
import ast
from scipy.spatial import kdtree

from simulation import Simulation
from evolve import Population

class Selection:
    def __init__(self, selection_type, additional_trials=False, simulation_settings=None):
        self.selection_type = selection_type
        self.additional_trials = additional_trials
        self.simulation_settings = simulation_settings
        self.archive = []
        self.novelty_scores = {}

        self.elite_size = None
        self.final_rounds = 8

    def select(self, population, n_individuals):
        trial_data = self.load_trial_data()
        self.update_individual_scores(population, trial_data)

        if self.selection_type == 'fitness':
            rankings = self.calculate_fitness_scores(population)
        elif self.selection_type == 'novelty':
            rankings = self.calculate_novelty_scores(population)
        elif self.selection_type == 'combined':
            rankings = self.calculate_combined_scores(population)

        top_individuals = self.select_top_individuals(population, rankings, n_individuals, self.selection_type)

        print("Top {} individuals:".format(n_individuals))
        for rank, individual in enumerate(top_individuals, start=1):
            print("Rank {}: {} ({})".format(rank, individual.name, individual.fitness))

        if self.additional_trials:
            top_individuals = self.run_additional_trials(top_individuals, self.final_rounds)

        print("Top {} individuals:".format(n_individuals))
        for rank, individual in enumerate(top_individuals, start=1):
            print("Rank {}: {} ({})".format(rank, individual.name, individual.fitness))
            print(f"Mutation History: {individual.mutation_history}")

        return top_individuals, rankings

    def run_additional_trials(self, selected_individuals, n_rounds):
        print(f"Running additional trials for {n_rounds} rounds")
        additional_trial_data = []
        
        for trial in range(n_rounds):
            print(f'Trial {trial} of {n_rounds}')
            for individual in selected_individuals:
                emitter_x = np.random.randint(900, 1100)
                if trial % 2:
                    emitter_y = np.random.randint(-200, -150)
                else:
                    emitter_y = np.random.randint(150, 200)
                trial_data_dict = Simulation(emitter_x, emitter_y, self.simulation_settings['neuron_type'], self.simulation_settings['recurrent']).simulate(
                    self.simulation_settings['wind_config'],
                    emitter_x,
                    emitter_y,
                    individual,
                    self.simulation_settings['neuron_type'],
                    self.simulation_settings['headless'],
                    self.simulation_settings['recurrent']
                )
                trial_data_dict['individual_name'] = individual.name
                additional_trial_data.append(trial_data_dict)
        
        additional_trial_data_df = pd.DataFrame(additional_trial_data)
        
        for individual in selected_individuals:
            individual_trials = additional_trial_data_df[additional_trial_data_df['individual_name'] == individual.name]
            
            if self.selection_type == 'fitness' or self.selection_type == 'combined':
                avg_fitness = individual_trials['fitness_score'].mean()
                individual.fitness = avg_fitness
            
            if self.selection_type == 'novelty' or self.selection_type == 'combined':
                trial_novelty_scores = [self.calculate_novelty_score(trial, additional_trial_data_df, self.archive) for _, trial in individual_trials.iterrows()]
                avg_novelty_score = sum(trial_novelty_scores) / len(trial_novelty_scores)
                individual.novelty_score = avg_novelty_score
        
        if self.selection_type == 'fitness':
            rankings = self.calculate_fitness_scores(selected_individuals)
        elif self.selection_type == 'novelty':
            rankings = self.calculate_novelty_scores(selected_individuals)
        elif self.selection_type == 'combined':
            rankings = self.calculate_combined_scores(selected_individuals)
        
        top_individuals = self.select_top_individuals(selected_individuals, rankings, len(selected_individuals), self.selection_type)
        return top_individuals

    def select_top_individuals(self, population, rankings, n_individuals, type):
        if type == 'fitness':
            sorted_indices = mx.argsort(rankings)
        else:
            sorted_indices = mx.argsort(rankings)[::-1]
        
        top_indices = sorted_indices[:n_individuals]
        top_individuals = [population.individuals[i] for i in top_indices.tolist()]
        
        return top_individuals

    def calculate_fitness_scores(self, population):
        fitness_scores = mx.array([individual.fitness for individual in population.individuals], dtype=mx.float32)
        return fitness_scores

    def calculate_novelty_scores(self, population):
        novelty_scores = mx.array([individual.novelty_score for individual in population.individuals], dtype=mx.float32)
        return novelty_scores

    def calculate_combined_scores(self, population, balance=0.5):
        fitness_scores = self.calculate_fitness_scores(population)
        novelty_scores = self.calculate_novelty_scores(population)

        min_fitness = mx.min(fitness_scores)
        max_fitness = mx.max(fitness_scores)
        normalized_fitness_scores = 1 - ((fitness_scores - min_fitness) / (max_fitness - min_fitness))

        min_novelty = mx.min(novelty_scores)
        max_novelty = mx.max(novelty_scores)
        normalized_novelty_scores = (novelty_scores - min_novelty) / (max_novelty - min_novelty)

        combined_scores = balance * normalized_fitness_scores + (1 - balance) * normalized_novelty_scores
        return combined_scores

    def calculate_novelty_score(self, trial, trial_data, archive):
        k = 15

        trial_record = mx.array(ast.literal_eval(trial['behaviour_record']), dtype=mx.float32)

        population_trials = trial_data[trial_data['individual_name'] != trial['individual_name']]
        population_records = mx.array([ast.literal_eval(record) for record in population_trials['behaviour_record']], dtype=mx.float32)

        if archive:
            archive_trials = pd.DataFrame(archive)
            archive_records = mx.array([ast.literal_eval(record) for record in archive_trials['behaviour_record']], dtype=mx.float32)
            total_records = mx.concatenate((population_records, archive_records))
        else:
            total_records = population_records

        # Create a KD Tree from the total records
        kd_tree = KDTree(total_records.reshape(total_records.shape[0], -1))

        # Query the KD Tree for the k nearest neighbors
        _, indices = kd_tree.query(trial_record.reshape(1, -1), k=k)

        # Retrieve the distances of the k nearest neighbors
        k_nearest_distances = mx.sqrt(mx.sum(mx.square(total_records[indices] - trial_record), axis=(1, 2)))
        novelty_score = mx.mean(k_nearest_distances)

        return novelty_score

    def calculate_distance(self, trial1, trial2):
        record1 = mx.array(trial1['behaviour_record'])
        record2 = mx.array(trial2['behaviour_record'])
        distance = mx.sqrt(mx.sum(mx.square(record2 - record1)))
        return distance

    def update_archive(self, trials):
        storage_chance = 0.02
        for trial in trials:
            if random.random() < storage_chance:
                archived_trial = {
                    'individual_name': trial['individual_name'],
                    'behaviour_record': trial['behaviour_record'],
                }
                self.archive.append(archived_trial)

    def update_trial_data(self, trial_data):
        if 'novelty_score' in trial_data.columns:
            trial_data['novelty_score'] = trial_data['novelty_score']
        
        if 'fitness_score' in trial_data.columns:
            trial_data['fitness_score'] = trial_data['fitness_score']
        
        with open('records/trial_data.csv', 'w', newline='') as csvfile:
            trial_data.to_csv(csvfile, index=False)

    def load_trial_data(self):
        with open('records/trial_data.csv', 'r') as csvfile:
            trial_data = pd.read_csv(csvfile)
        return trial_data

    def update_individual_scores(self, individuals, trial_data):
        if isinstance(individuals, Population):
            individuals = individuals.individuals

        for individual in individuals:
            individual_trials = trial_data[trial_data['individual_name'].astype(str) == str(individual.name)]
            
            if 'fitness_score' in individual_trials.columns:
                avg_fitness = individual_trials['fitness_score'].mean()
                individual.fitness = avg_fitness

            if 'novelty_score' in individual_trials.columns:
                trial_novelty_scores = [self.calculate_novelty_score(trial, trial_data, self.archive) for _, trial in individual_trials.iterrows()]
                avg_novelty_score = sum(trial_novelty_scores) / len(trial_novelty_scores)
                individual.novelty_score = avg_novelty_score