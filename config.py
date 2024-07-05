CONFIG = {
    'training': {
        'mode': 'new',  # 'new' for training a new network, 'load' for loading a saved network
        'load_file': './records/latest_network.npz',  # File to load if mode is 'load'
        'save_file': './records/latest_network.npz',  # File to save the network to after each epoch
    },
    'network': {
        'feedforward': {
            'n_in': 100,
            'n_hidden': 50,
            'n_layers': 2,
            'n_out': 3,
        },
        'recurrent': {
            'n_receptors': 10,
            'n_hidden': 50,
            'n_layers': 4,
            'n_out': 3,
            'sequence_n': 12,
        },
        'spiking': {
        },
        'reservoir': {
        }
    },
    'method': {
        'evolution': {
            'population_size': 50,
            'mutation_rate': 0.1,
        },
        'q_learning': {
            'learning_rate': 0.001,
            'gamma': 0.975,
            'epsilon': 0.1,
            'n_actions': 3,
        },
        'policy_gradient': {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'n_actions': 3,
        },
        'actor_critic': {
            'learning_rate': 0.0002,
            'gamma': 0.99,
            'n_actions': 3,
            'critic_coefficient': 0.5,
            'entropy_coefficient': 0.01,
            'max_grad_norm': 0.5,
        },
    },
    'simulation': {
        'headless': False,
        'processes': 1,
        'max_steps': 1000,
        'screen_size': (1400, 1000),
    }
}