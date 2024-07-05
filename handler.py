import mlx.core as mx
import mlx.optimizers as opt
import mlx.nn as nn
import os
from networks.ann.network import Network
from networks.ann.recurrent import RecurrentNetwork
from methods.learning import q_learning, policy_gradient, actor_critic

class Handler:
    def __init__(self, network_type, network_params, method_type, method_params):
        self.network_type = network_type
        self.method = self._init_method(method_type, method_params)
        self.network = self._init_network(network_type, network_params, method_type)
        self.last_state = None
        self.last_action = None
        self.state_stack = []
        self.sequence_n = network_params.get('sequence_n', 1)
        self.episode_count = 1

        self.environment = None
        self.boundary_punishment = 0

    def _init_network(self, network_type, network_params, method_type):
        # Adjust n_out based on the method type
        if method_type == 'actor_critic':
            # For actor-critic, we need two outputs: action logits and value
            network_params['n_out'] = (network_params['n_out'], 1)
        
        if network_type == 'feedforward':
            return Network(**network_params)
        elif network_type == 'recurrent':
            return RecurrentNetwork(**network_params)
        else:
            raise ValueError(f"Invalid network type: {network_type}")

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
        
        else:
            raise ValueError(f"Invalid method type: {method_type}")

    def _check_required_params(self, params, required):
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")

    def get_action(self, state):
        if self.network_type == 'recurrent':
            self.state_stack = self.roll_and_update(self.state_stack, mx.array(state))
            state_input = mx.stack(self.state_stack)
            state_input = mx.expand_dims(state_input, axis=0)  # Shape: (1, sequence_n, n_receptors)
        else:
            state_input = mx.expand_dims(mx.array(state), axis=0)  # Shape: (1, n_receptors)

        if self.method == 'q_learning':
            epsilon = self.epsilon * (1 / (1 + self.episode_count * 0.01))
            action = q_learning.epsilon_greedy_action(self.network, state_input, epsilon, self.n_actions)
        elif self.method == 'policy_gradient':
            action = policy_gradient.select_action(self.network, state_input)
        elif self.method == 'actor_critic':
            action = actor_critic.select_action(self.network, state_input)
        else:
            raise ValueError(f"Invalid method: {self.method}")
        
        self.last_state = state
        self.last_action = action
        return action

    def update(self, reward, new_state, done):
        if reward != None:
            reward -= self.boundary_punishment
        else:
            reward = -self.boundary_punishment

        if self.method == 'q_learning':
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
        elif self.method in ['policy_gradient', 'actor_critic']:
            self.episode_states.append(self.last_state)
            self.episode_actions.append(self.last_action)
            self.episode_rewards.append(reward)
            
            if done:
                states = mx.stack([mx.array(s) for s in self.episode_states])
                actions = mx.array(self.episode_actions)
                rewards = mx.array(self.episode_rewards)
                
                if self.network_type == 'recurrent':
                    states = mx.stack([mx.stack(self.roll_and_update([], s)) for s in states])
                
                if self.method == 'policy_gradient':
                    loss = policy_gradient.update_policy_network(
                        self.network,
                        self.optimizer,
                        states,
                        actions,
                        rewards,
                        self.gamma
                    )
                elif self.method == 'actor_critic':
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
        
        if done:
            self.state_stack = []
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
        self.environment = None

    def update_environment(self, environment, boundary_punishment):
        self.environment = environment
        self.boundary_punishment = boundary_punishment

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.network.save_weights(path)
        print(f"Model weights saved to {path}")

    def load(self, path):
        self.network.load_weights(path)
        print(f"Model weights loaded from {path}")

class FitnessTracker:
    def __init__(self):
        self.run_id = str(ULID())
        self.csv_path = f"./records/fitness_data_{self.run_id}.csv"
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