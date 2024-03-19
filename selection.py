# This will handle all variants of selections: fitness, novelty, etc.
# It will also handle the subprotocols such as final rounds for elitism
import numpy as np
import pandas as pd
import random

class Selection:
    def __init__(self, selection_type):
        self.selection_type = selection_type
        self.archive = []
        self.novelty_scores = {}

        self.elite_size = None
        self.final_rounds = 20

    def select(self, population, n_individuals):
        # Select the top n individuals based on fitness or novelty
        if self.selection_type == 'fitness':
            rankings = self.calculate_fitness(population)
            top_individuals = rankings[:n_individuals]
        elif self.selection_type == 'novelty':
            rankings = self.calculate_novelty(population)
            top_individuals = rankings[:n_individuals]

        print("Top {} individuals:".format(n_individuals))
        for rank, individual in enumerate(top_individuals, start=1):
            print("Rank {}: {}".format(rank, individual.name))

        return top_individuals

    def calculate_fitness(self, population):
        # Load trial data from CSV
        trial_data = pd.read_csv('records/trial_data.csv')

        # Group trial data by individual
        grouped_data = trial_data.groupby('individual_name')

        fitness_scores = {}
        for individual_name, group in grouped_data:
            # Calculate average fitness across trials
            avg_fitness = group['score'].mean()
            fitness_scores[individual_name] = avg_fitness

        # Sort individuals by fitness scores
        sorted_individuals = sorted(population.individuals, key=lambda x: fitness_scores.get(x.name, float('-inf')), reverse=True)
        print(f'Sorted individuals: {sorted_individuals}')
        return sorted_individuals

    def additional_trials(self, population, n_individuals, n_trials):
        # Select the top n individuals based on fitness
        top_individuals = self.calculate_fitness(population)[:n_individuals]

        # Run additional trials for the top individuals
        additional_trial_data = []
        for individual in top_individuals:
            for _ in range(n_trials):
                # Simulate additional trials and collect data
                trial_data = Simulation.simulate(individual)
                additional_trial_data.append(trial_data)

        # Calculate average fitness for the additional trials
        additional_fitness_scores = {}
        for individual in top_individuals:
            individual_trials = [trial for trial in additional_trial_data if trial['individual_name'] == individual.name]
            avg_fitness = sum(trial['score'] for trial in individual_trials) / len(individual_trials)
            additional_fitness_scores[individual.name] = avg_fitness

        # Reorder the top individuals based on their additional trial performance
        reordered_individuals = sorted(top_individuals, key=lambda x: additional_fitness_scores.get(x.name, float('-inf')), reverse=True)

        return reordered_individuals

    def calculate_novelty(self, population):
        # Load trial data from CSV
        with open('records/trial_data.csv', 'r') as csvfile:
            trial_data = pd.read_csv(csvfile)
        
        # Iterate over each individual in the population
        for individual in population.individuals:
            
            individual_trials = trial_data[trial_data['individual_name'].astype(str) == str(individual.name)]
            
            # Calculate novelty score for each trial of the individual
            for index, trial in individual_trials.iterrows():
                trial_novelty_score = self.calculate_novelty_score(trial, trial_data, self.archive)
                trial_data.at[index, 'score'] = trial_novelty_score
                trial_data.at[index, 'type'] = 'novelty'
        
        # Update the archive with the most novel trials
        self.update_archive(trial_data.to_dict('records'))

        # Update the trial data with novelty scores using the separate method
        self.update_trial_data_with_novelty_scores(trial_data)
        
        novelty_scores = {}
        for individual in population.individuals:
            individual_trials = trial_data[trial_data['individual_name'] == individual.name]
            if not individual_trials.empty:
                # Calculate the novelty scores for each trial
                trial_novelty_scores = individual_trials.apply(lambda row: self.calculate_novelty_score(row, trial_data, self.archive), axis=1)
                
                # # By mean
                # mean_novelty_score = trial_novelty_scores.mean()
                # novelty_scores[individual.name] = mean_novelty_score
                
                # # By maximum
                # max_novelty_score = trial_novelty_scores.max()
                # novelty_scores[individual.name] = max_novelty_score
                
                # By minimum
                min_novelty_score = trial_novelty_scores.min()
                novelty_scores[individual.name] = min_novelty_score
        
        # Sort individuals by overall novelty scores
        sorted_individuals = sorted(population.individuals, key=lambda x: novelty_scores.get(x.name, float('-inf')), reverse=True)
        
        return sorted_individuals

    def calculate_novelty_score(self, trial, trial_data, archive):
        # Calculate the novelty score based on the distance from other individuals in the population and a subset of the archive
        population_distances = [self.calculate_distance(trial, other_trial) for _, other_trial in trial_data.iterrows() if other_trial['individual_name'] != trial['individual_name']]
        
        # Randomly sample a subset of the archive for comparison
        archive_subset = random.sample(archive, min(len(archive), 100))  # Adjust the sample size as needed
        archive_distances = [self.calculate_distance(trial, archived_trial) for archived_trial in archive_subset if archived_trial['individual_name'] != trial['individual_name']]
        
        total_distances = population_distances + archive_distances
        if total_distances:
            novelty_score = sum(total_distances) / len(total_distances)
        else:
            novelty_score = 0
        
        return novelty_score

    def calculate_distance(self, trial1, trial2):
        # Calculate the Euclidean distance between two trials based on their final positions
        x1 = trial1['final_x']
        y1 = trial1['final_y']
        x2 = trial2['final_x']
        y2 = trial2['final_y']
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def update_archive(self, trials):
        # Update the archive with the most novel trials
        # sorted_trials = sorted(trials, key=lambda x: x['novelty_score'], reverse=True)
        # self.archive = sorted_trials[:len(trials)//2]
        # print("Updated archive:")
        # for trial in self.archive:
        #     print(f"Trial {trial['individual_name']} (Epoch {trial['epoch_number']}, Trial {trial['trial_number']}), Novelty Score: {trial['novelty_score']}")

        # Update the archive with trials based on a percentage chance
        storage_chance = 0.01
        for trial in trials:
            if random.random() < storage_chance:
                # Store only the essential information in the archive
                archived_trial = {
                    'individual_name': trial['individual_name'],
                    'final_x': trial['final_x'],
                    'final_y': trial['final_y']
                }
                self.archive.append(archived_trial)

    def update_trial_data_with_novelty_scores(self, trial_data):
        # Update the trial data with novelty scores
        trial_data['score'] = trial_data['score']
        trial_data['type'] = 'novelty'
        
        # Save the updated trial data back to CSV
        with open('records/trial_data.csv', 'w', newline='') as csvfile:
            trial_data.to_csv(csvfile, index=False)