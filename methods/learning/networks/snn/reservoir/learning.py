import mlx.core as mx
import mlx.nn as nn

class ActorCriticLearning:
    def __init__(self, n_res, n_act, n_pred=1, learning_rate=0.01, gamma=0.99, steps=10):
        self.n_res = n_res
        self.n_act = n_act
        self.n_pred = n_pred
        self.lr = learning_rate
        self.gamma = gamma
        self.steps = steps
        
        self.weights = mx.random.normal((n_res, n_act + n_pred)) * 0.01
        
        # Temporal decay factors
        self.temporal_factors = mx.array([1.0 - 0.09 * i for i in range(steps)])
        
        self.prev_value = None
        self.prev_spike_history = None
        self.prev_action = None

    def learn(self, spike_history, action, reward):
        # spike_history shape: (steps, n_res, n_act)
        action_values, state_value = self.forward(spike_history)
        
        if self.prev_value is not None:
            td_error = reward + self.gamma * state_value - self.prev_value
            
            # Calculate the contribution of each reservoir neuron to the action
            weighted_spikes = spike_history * self.temporal_factors[:, mx.newaxis, mx.newaxis]
            reservoir_contribution = mx.sum(weighted_spikes, axis=0)  # (n_res, n_act)
            
            # Update action weights
            actor_gradient = td_error * reservoir_contribution[:, action]
            self.weights[:, :self.n_act] += self.lr * actor_gradient.reshape(-1, 1)
            
            # Update critic weights
            critic_gradient = td_error * mx.sum(reservoir_contribution, axis=1)
            self.weights[:, -self.n_pred:] += self.lr * critic_gradient.reshape(-1, 1)
        
        self.prev_value = state_value
        self.prev_spike_history = spike_history
        self.prev_action = action
        
        print(f"Updated weights (first few): {self.weights[:5, :]}")

    def get_weights(self):
        return self.weights

    def reset(self):
        self.prev_value = None
        self.prev_spike_history = None
        self.prev_action = None