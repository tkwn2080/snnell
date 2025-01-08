import caffeine

import multiprocessing
import pygame
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from config import CONFIG

from handler import Handler

import csv
from ulid import ULID
import time
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Select network and method types
    network_type = 'spiking'
    method_type = 'NEAT'

    # Load default configurations
    network_params = CONFIG['network'][network_type].copy()
    method_params = CONFIG['method'][method_type].copy()
    sim_params = CONFIG['simulation'].copy()
    training_params = CONFIG['training'].copy()

    # Override parameters if needed
    network_params.update({
        # Add other network parameter overrides here
    })

    method_params.update({
        # Add other method parameter overrides here
    })

    sim_params.update({
        # 'headless': False,
        # 'processes': 1,
        # Add other simulation parameter overrides here
    })

    # Create Environment
    if not sim_params['headless']:
        pygame.init()
        screen = pygame.display.set_mode(sim_params['screen_size'], pygame.DOUBLEBUF | pygame.HWSURFACE)
        clock = pygame.time.Clock()
        environment = screen, clock
    else:
        environment = None

    # Initialise Handler
    handler = Handler(network_type, network_params, method_type, method_params, sim_params, environment)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()