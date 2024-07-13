import mlx.core as mx
import mlx.optimizers as opt
import mlx.nn as nn
import os
from networks.ann.network import Network
from networks.ann.recurrent import RecurrentNetwork
from methods.learning import q_learning, policy_gradient, actor_critic, soft_ac

class LearningController:
    def __init__(self, network_type, network_params, method_type, method_params):
        self.network_type = network_type
        self.method_type = method_type
        self.network = self._init_network(network_type, network_params, method_type, method_params)
        self.method = self._init_method(method_type, method_params)
        self.last_state = None
        self.last_action = None
        self.state_stack = []
        self.sequence_n = network_params.get('sequence_n', 1)
        self.episode_count = 1
        self.is_sequence_full = False

        self.environment = None
        self.boundary_punishment = 0

        self.action_type = 'continuous' if method_type == 'soft_ac' else 'discrete'

    def _init_network(self, network_type, network_params, method_type, method_params):
        if method_type == 'q_learning':
            n_out = method_params['n_actions']
        elif method_type == 'policy_gradient':
            n_out = method_params['n_actions']
        elif method_type == 'actor_critic':
            n_out = (method_params['n_actions'], 1)
        elif method_type == 'soft_ac':
            n_out = method_params['action_dim'] 
        else:
            raise ValueError(f"Invalid method type: {method_type}")

        network_params['n_out'] = n_out
        network_params['method_type'] = method_type
        
        if network_type == 'feedforward':
            network = Network(**network_params)
        elif network_type == 'recurrent':
            network = RecurrentNetwork(**network_params)
        else:
            raise ValueError(f"Invalid network type: {network_type}")
        
        return network

    def _init_method(self, method_type, method_params):
        if method_type == 'q_learning':
            required_params = ['learning_rate', 'gamma', 'epsilon', 'n_actions']
            self._check_required_params(method_params, required_params)
            
            self.optimizer = opt.Adam(learning_rate=method_params['learning_rate'])
            self.gamma = method_params['gamma']
            self.epsilon = method_params['epsilon']
            self.n_actions = method_params['n_actions']
            return 'q_learning'
        
        elif method_type == 'policy_gradient':
            required_params = ['learning_rate', 'gamma', 'n_actions']
            self._check_required_params(method_params, required_params)
            
            self.optimizer = opt.Adam(learning_rate=method_params['learning_rate'])
            self.gamma = method_params['gamma']
            self.n_actions = method_params['n_actions']
            self.episode_states = []
            self.episode_actions = []
            self.episode_rewards = []
            return 'policy_gradient'
        
        elif method_type == 'actor_critic':
            required_params = ['learning_rate', 'gamma', 'n_actions', 'max_grad_norm']
            self._check_required_params(method_params, required_params)
            
            self.optimizer = opt.Adam(learning_rate=method_params['learning_rate'])
            self.gamma = method_params['gamma']
            self.n_actions = method_params['n_actions']
            self.max_grad_norm = method_params['max_grad_norm']
            self.episode_states = []
            self.episode_actions = []
            self.episode_rewards = []
            return 'actor_critic'
        
        elif method_type == 'soft_ac':
            required_params = ['learning_rate', 'gamma', 'alpha', 'tau', 'batch_size', 'buffer_size', 'action_dim', 'state_dim', 'hidden_dim']
            self._check_required_params(method_params, required_params)
            
            method = soft_ac.SoftActorCritic(
                self.network,
                soft_ac.CriticNetwork(method_params['state_dim'], method_params['action_dim'], method_params['hidden_dim']),
                method_params['state_dim'],
                method_params['action_dim'],
                method_params['buffer_size'],
                method_params['batch_size'],
                method_params['gamma'],
                method_params['alpha'],
                method_params['tau'],
                method_params['learning_rate'],
                method_params['sequence_n'],
                method_params['use_fractional_updates'],
                method_params['update_steps']
            )
            return method
        
        else:
            raise ValueError(f"Invalid method type: {method_type}")

    def _check_required_params(self, params, required):
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")

    def get_action(self, state):
        if self.network_type == 'recurrent':
            if not self.is_sequence_full:
                self.state_stack.append(mx.array(state))
                if len(self.state_stack) == self.sequence_n:
                    self.is_sequence_full = True
                else:
                    return mx.zeros(self.method.action_dim if self.method_type == 'soft_ac' else self.n_actions)
            else:
                self.state_stack = self.state_stack[1:] + [mx.array(state)]

            state_input = mx.stack(self.state_stack)
            state_input = mx.expand_dims(state_input, axis=0)
        else:
            state_input = mx.expand_dims(mx.array(state), axis=0)

        if self.method_type == 'q_learning':
            epsilon = self.epsilon * (1 / (1 + self.episode_count * 0.01))
            action = q_learning.epsilon_greedy_action(self.network, state_input, epsilon, self.n_actions)
        elif self.method_type == 'policy_gradient':
            action = policy_gradient.select_action(self.network, state_input)
        elif self.method_type == 'actor_critic':
            action = actor_critic.select_action(self.network, state_input)
        elif self.method_type == 'soft_ac':
            action = self.method.select_action(state_input)
        else:
            raise ValueError(f"Invalid method: {self.method_type}")
        
        self.last_state = state
        self.last_action = action
        return action

    def update(self, reward, new_state, done):
        if reward is not None:
            reward -= self.boundary_punishment
        else:
            reward = -self.boundary_punishment

        if self.network_type == 'recurrent' and not self.is_sequence_full:
            return None

        if self.method_type == 'q_learning':
            if self.last_state is not None and self.last_action is not None:
                if self.network_type == 'recurrent':
                    last_state_input = mx.expand_dims(mx.stack(self.state_stack), axis=0)
                    new_state_stack = self.roll_and_update(self.state_stack, mx.array(new_state))
                    new_state_input = mx.expand_dims(mx.stack(new_state_stack), axis=0)
                else:
                    last_state_input = mx.expand_dims(mx.array(self.last_state), axis=0)
                    new_state_input = mx.expand_dims(mx.array(new_state), axis=0)
                
                loss = q_learning.update_q_network(
                    self.network,
                    self.optimizer,
                    last_state_input,
                    self.last_action,
                    reward,
                    new_state_input,
                    done,
                    self.gamma
                )
            else:
                loss = None
        elif self.method_type in ['policy_gradient', 'actor_critic']:
            self.episode_states.append(self.last_state)
            self.episode_actions.append(self.last_action)
            self.episode_rewards.append(reward)
            
            if done:
                states = mx.stack([mx.array(s) for s in self.episode_states])
                actions = mx.array(self.episode_actions)
                rewards = mx.array(self.episode_rewards)
                
                if self.network_type == 'recurrent':
                    states = mx.stack([mx.stack(self.roll_and_update([], s)) for s in states])
                
                if self.method_type == 'policy_gradient':
                    loss = policy_gradient.update_policy_network(
                        self.network,
                        self.optimizer,
                        states,
                        actions,
                        rewards,
                        self.gamma
                    )
                elif self.method_type == 'actor_critic':
                    loss, actor_loss, critic_loss, entropy_loss = actor_critic.update_actor_critic_network(
                        self.network,
                        self.optimizer,
                        states,
                        actions,
                        rewards,
                        self.gamma,
                        self.max_grad_norm
                    )
                
                self.episode_states = []
                self.episode_actions = []
                self.episode_rewards = []
            else:
                loss = None
        elif self.method_type == 'soft_ac':
            if self.last_state is not None and self.last_action is not None:
                loss = self.method.update(self.last_state, self.last_action, reward, new_state, done)
            else:
                loss = None
        
        if done:
            self.state_stack = []
            self.is_sequence_full = False
            self.done()
        
        return loss

    def roll_and_update(self, stack, new_value):
        if len(stack) < self.sequence_n:
            stack.append(new_value)
        else:
            stack = stack[1:] + [new_value]
        return stack
    
    def done(self):
        self.episode_count += 1
        self.reset()

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.state_stack = []
        self.is_sequence_full = False
        self.environment = None

    def update_environment(self, environment, boundary_punishment):
        self.environment = environment
        self.boundary_punishment = boundary_punishment

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.method_type == 'soft_ac':
            self.method.save(path)
        else:
            self.network.save_weights(path)

    def load(self, path):
        if self.method_type == 'soft_ac':
            self.method.load(path)
        else:
            self.network.load_weights(path)

    def get_action_type(self):
        return self.action_type



# This is from the previous index.py, before modifications for addition of evolutionary methods

#     # Load previous network state if specified
#     if training_params['mode'] == 'load':
#         load_path = training_params['load_file']
#         if os.path.exists(load_path):
#             handler.load(load_path)
#             print(f"Loaded previous network state from {load_path}")
#         else:
#             print(f"Warning: Specified load file {load_path} not found. Starting with a new network.")

#     # Ensure save_path is valid
#     save_path = os.path.join(os.getcwd(), training_params['save_file'])

#     # Create FitnessTracker
#     fitness_tracker = FitnessTracker()

#     # Create and run Simulation
#     for epoch_n in range(epochs):
#         config = SimulationConfig(epoch_n)
        
#         # Merge simulation_params into config
#         for key, value in simulation_params.items():
#             setattr(config, key, value)
        
#         simulation = Simulation(config, handler)
#         results = simulation.run(headless=simulation_params['headless'], environment=environment)

#         # Calculate fitness
#         fitness = calculate_fitness(results)

#         # Record fitness
#         fitness_tracker.record_fitness(epoch_n, fitness)

#         # Process results
#         print(f"Epoch {epoch_n + 1}/{epochs}: Fitness = {fitness}")

#         # Save network state after each epoch
#         handler.save(save_path)
#         print(f"Saved network state to {save_path}")

#     # Plot fitness at the end of the run
#     fitness_tracker.plot_fitness()

#     print("Training completed.")

# def calculate_fitness(results):
#     final_pos = np.array(results['final_position'])
#     emitter_pos = np.array(results['emitter_position'])
    
#     # Calculate non-normalized distance
#     distance = np.linalg.norm(final_pos - emitter_pos)
    
#     if results['collided']:
#         print(f'Collision: {distance}')
#         return 0  # Minimum distance (best fitness) when collision occurs
#     else:
#         print(f'No collision: {distance}')
#         return distance  # Return the non-normalized Euclidean distance

# class FitnessTracker:
#     def __init__(self):
#         self.run_id = str(ULID())
#         self.csv_path = f"./records/fd_at_{time.strftime('%Y-%m-%d_%H-%M')}_{self.run_id}.csv"
#         self.fitness_data = []
        
#         with open(self.csv_path, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(["Epoch", "Fitness"])
    
#     def record_fitness(self, epoch, fitness):
#         self.fitness_data.append((epoch, fitness))
#         with open(self.csv_path, 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([epoch, fitness])
    
#     def plot_fitness(self, window_size=10):
#         epochs, fitness = zip(*self.fitness_data)
        
#         # Apply windowed smoothing
#         smoothed_fitness = np.convolve(fitness, np.ones(window_size)/window_size, mode='valid')
#         smoothed_epochs = epochs[window_size-1:]
        
#         plt.figure(figsize=(12, 6))
#         plt.plot(epochs, fitness, label='Raw Fitness', alpha=0.5)
#         plt.plot(smoothed_epochs, smoothed_fitness, label=f'Smoothed Fitness (window={window_size})')
#         plt.xlabel('Epoch')
#         plt.ylabel('Fitness')
#         plt.title(f'Fitness over Training Run {self.run_id}')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(f"./records/fitness_plot_at_{time.strftime('%Y-%m-%d_%H-%M')}_{self.run_id}.png")
#         plt.close()