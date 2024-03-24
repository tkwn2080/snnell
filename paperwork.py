import csv
import string
import numpy as np
import random
import os
import shutil
import datetime

# PAPERWORK
class Paperwork: 
    def collate_genotype(genotype):
        # genotype_str = f"Learning Rate: {genotype[8][0]}, Eligibility Decay: {genotype[8][1]}, Recurrent Layer: {genotype[9]}"
        weights_str = genotype[5]
        return weights_str

    def epoch_csv(epoch_data, epoch, architecture):
        # Determine if the file needs a header by checking its existence or size
        # epoch_csv_filename = './records/' + time.strftime("%d%m%Y%H") + '_epoch_data.csv'
        epoch_csv_filename = './records/epoch_data.csv'
        write_header = not os.path.exists(epoch_csv_filename) or os.path.getsize(epoch_csv_filename) == 0
        with open(epoch_csv_filename, 'a', newline='') as csvfile:
            fieldnames = ['epoch', 'name', 'seed', 'fitness', 'architecture', 'heritage']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader()

            # print(f'Epoch data: {epoch_data}')

            for data in epoch_data:
                writer.writerow({
                    'epoch': epoch + 1,
                    'name': data['name'],
                    'seed': data['seed'],
                    'fitness': data['avg_fitness'],
                    'architecture': architecture,
                    'heritage': data['heritage'],
                })

    def calculate_fitness(simulation_data):
        emitter_x, emitter_y = simulation_data['emitter_position']
        if simulation_data['collided']:
            collision_time = simulation_data['collision_time']
            return collision_time * collision_time
        else:
            final_x, final_y = simulation_data['final_position']
            simulation_time = simulation_data['simulation_time']
            final_distance = np.hypot(final_x - emitter_x, final_y - emitter_y)
            return (final_distance * final_distance) / 100

    def trial_csv(trial_data):
        trial_csv_filename = './records/trial_data.csv'
        trial_fieldnames = ['epoch_number', 'trial_number', 'individual_name', 'novelty_score', 'fitness_score', 'final_x', 'final_y', 'emitter_x', 'emitter_y', 'behaviour_record']
        write_header = not os.path.exists(trial_csv_filename) or os.path.getsize(trial_csv_filename) == 0

        with open(trial_csv_filename, 'a', newline='') as trial_csvfile:
            trial_writer = csv.DictWriter(trial_csvfile, fieldnames=trial_fieldnames)
            if write_header:
                trial_writer.writeheader()
            trial_writer.writerow(trial_data)
    
    def generate_random_name(length=6):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))

    def move_existing_files():
        record_dir = './records/'
        trial_data_file = os.path.join(record_dir, 'trial_data.csv')
        epoch_data_file = os.path.join(record_dir, 'epoch_data.csv')

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        destination_folder = os.path.join(record_dir, timestamp)

        if os.path.exists(trial_data_file) or os.path.exists(epoch_data_file):
            os.makedirs(destination_folder, exist_ok=True)

            if os.path.exists(trial_data_file):
                shutil.move(trial_data_file, os.path.join(destination_folder, "trial_data.csv"))

            if os.path.exists(epoch_data_file):
                shutil.move(epoch_data_file, os.path.join(destination_folder, "epoch_data.csv"))