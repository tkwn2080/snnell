import os
import multiprocessing
import pygame
from config import CONFIG
from handler import Handler
from simulation.simulation import Simulation
from simulation.sim_config import SimulationConfig

import csv
from ulid import ulid as ULID
import time
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Select network and method types
    network_type = 'recurrent'
    method_type = 'actor_critic'
    epochs = 1000

    # Load default configurations
    network_params = CONFIG['network'][network_type].copy()
    method_params = CONFIG['method'][method_type].copy()
    simulation_params = CONFIG['simulation'].copy()
    training_params = CONFIG['training'].copy()

    # Override parameters if needed
    network_params.update({
        # Add other network parameter overrides here
    })

    method_params.update({
        # Add other method parameter overrides here
    })

    simulation_params.update({
        # 'headless': True,
        # 'processes': 2,
        # Add other simulation parameter overrides here
    })

    # Create Environment
    pygame.init()
    screen = pygame.display.set_mode(simulation_params['screen_size'], pygame.DOUBLEBUF | pygame.HWSURFACE)
    clock = pygame.time.Clock()
    environment = screen, clock

    # Create Handler
    handler = Handler(network_type, network_params, method_type, method_params)

    # Load previous network state if specified
    if training_params['mode'] == 'load':
        load_path = training_params['load_file']
        if os.path.exists(load_path):
            handler.load(load_path)
            print(f"Loaded previous network state from {load_path}")
        else:
            print(f"Warning: Specified load file {load_path} not found. Starting with a new network.")

    # Ensure save_path is valid
    save_path = os.path.join(os.getcwd(), training_params['save_file'])

    # Create FitnessTracker
    fitness_tracker = FitnessTracker()

    # Create and run Simulation
    for epoch_n in range(epochs):
        config = SimulationConfig(epoch_n)
        
        # Merge simulation_params into config
        for key, value in simulation_params.items():
            setattr(config, key, value)
        
        simulation = Simulation(config, handler)
        results = simulation.run(headless=simulation_params['headless'], environment=environment)

        # Calculate fitness (you might need to adjust this based on your specific fitness criteria)
        fitness = calculate_fitness(results)

        # Record fitness
        fitness_tracker.record_fitness(epoch_n, fitness)

        # Process results
        print(f"Epoch {epoch_n + 1}/{epochs}: Fitness = {fitness}")

        # Save network state after each epoch
        handler.save(save_path)
        print(f"Saved network state to {save_path}")

    # Plot fitness at the end of the run
    fitness_tracker.plot_fitness()

    print("Training completed.")







def calculate_fitness(results):
    final_pos = np.array(results['final_position'])
    emitter_pos = np.array(results['emitter_position'])
    
    # Calculate non-normalized distance
    distance = np.linalg.norm(final_pos - emitter_pos)
    
    if results['collided']:
        print(f'Collision: {distance}')
        return 0  # Minimum distance (best fitness) when collision occurs
    else:
        print(f'No collision: {distance}')
        return distance  # Return the non-normalized Euclidean distance

class FitnessTracker:
    def __init__(self):
        self.run_id = str(ULID())
        self.csv_path = f"./records/fd_at_{time.strftime('%Y-%m-%d_%H-%M')}_{self.run_id}.csv"
        self.fitness_data = []
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Fitness"])
    
    def record_fitness(self, epoch, fitness):
        self.fitness_data.append((epoch, fitness))
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, fitness])
    
    def plot_fitness(self, window_size=10):
        epochs, fitness = zip(*self.fitness_data)
        
        # Apply windowed smoothing
        smoothed_fitness = np.convolve(fitness, np.ones(window_size)/window_size, mode='valid')
        smoothed_epochs = epochs[window_size-1:]
        
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, fitness, label='Raw Fitness', alpha=0.5)
        plt.plot(smoothed_epochs, smoothed_fitness, label=f'Smoothed Fitness (window={window_size})')
        plt.xlabel('Epoch')
        plt.ylabel('Fitness')
        plt.title(f'Fitness over Training Run {self.run_id}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"fitness_plot_{self.run_id}.png")
        plt.close()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()