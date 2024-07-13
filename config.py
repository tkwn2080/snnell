CONFIG = {
    'training': {
        'mode': 'new',  # 'new' for training a new network, 'load' for loading a saved network
        'load_file': './records/latest_network.npz',  # File to load if mode is 'load'
        'save_file': './records/latest_network.npz',  # File to save the network to after each epoch
    },
    'network': {
        'feedforward': {
            'n_in': 100,
            'n_hidden': 64,  # Increased for more complex tasks
            'n_layers': 2,
            'n_out': 3,  # This will be overwritten based on the method
        },
        'recurrent': {
            'n_receptors': 10,
            'n_hidden': 64,  # Increased for more complex tasks
            'n_layers': 2,
            'n_out': 1,  # This will be overwritten based on the method
            'sequence_n': 12,
        },
        'spiking': {
            'n_in': 10,
            'n_out': 3,
            'input_duration': 20,
            'input_current': 40,
            'propagation_steps': 40,
            'action_type': 'continuous',
        },
        'reservoir': {
            # Add reservoir network parameters if needed
        }
    },
    'method': {
        'NEAT': {
            'population_size': 150,
            'n_generations': 100,
            'n_trials': 4,
            'NEAT_parameters': {
                "speciation": {
                "c_one": 1.1,
                "c_two": 1.1,
                "c_three": 0.5,
                "compatibility_threshold": 1.4,
                },
                "mutation": {
                    "weight_mutation_rate": 0.8,
                    "weight_perturbation_rate": 0.9, 
                    "connection_mutation_rate": 0.1,
                    "node_mutation_rate": 0.06
                },
                "reproduction": {
                    "reproduction_rate": 0.75,
                    "elimination_rate": 0.1,
                    "interspecies_mating_rate": 0.0,  
                }
            }
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
        'soft_ac': {
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'alpha': 0.2,  # Temperature parameter
            'tau': 0.005,  # Soft update coefficient
            'batch_size': 64,
            'buffer_size': 1000000,
            'action_dim': 2,  # Dimension of the action space (e.g., 2 for [movement, rotation])
            'state_dim': 10,  # Dimension of the state space (should match n_in)
            'hidden_dim': 64,  # Hidden layer size for the critic network
            'n_actions': 2,  # For compatibility with other methods, represents action_dim here
            'sequence_n': 12,  # Hardcoded for now, should be inferred from network parameters
            'use_fractional_updates': False,  # Whether to use fractional updates, false or it doesn't work
            'update_steps': 10,  # Number of gradient steps to take per update
        },
    },
    'simulation': {
        'headless': True,
        'processes': 10,
        'max_steps': 100,
        'screen_size': (1400, 1000),
        'visualize_best': True,
        'n_best_genomes': 3
    }
}