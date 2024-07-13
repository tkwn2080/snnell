import mlx.core as mx
import mlx.nn as nn

class Network(nn.Module):
    def __init__(self, n_in, n_hidden, n_layers, n_out, method_type='q_learning'):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.method_type = method_type
        
        # Input layer
        self.layers = [nn.Linear(n_in, n_hidden)]
        
        # Hidden layers
        self.layers.extend([nn.Linear(n_hidden, n_hidden) for _ in range(n_layers - 1)])
        
        # Output layer(s)
        if method_type == 'actor_critic':
            self.actor_head = nn.Linear(n_hidden, n_out[0])
            self.critic_head = nn.Linear(n_hidden, n_out[1])
        elif method_type == 'soft_ac':
            self.actor_mean = nn.Linear(n_hidden, n_out)
            self.actor_log_std = nn.Linear(n_hidden, n_out)
        else:  # Q-learning or Policy Gradient
            self.layers.append(nn.Linear(n_hidden, n_out))
        
        # Initialize weights and biases
        for layer in self.layers:
            nn.init.he_normal()(layer.weight)
            nn.init.constant(0.01)(layer.bias)
        
        if hasattr(self, 'actor_head'):
            nn.init.he_normal()(self.actor_head.weight)
            nn.init.constant(0.01)(self.actor_head.bias)
            nn.init.he_normal()(self.critic_head.weight)
            nn.init.constant(0.01)(self.critic_head.bias)
        elif hasattr(self, 'actor_mean'):
            nn.init.he_normal()(self.actor_mean.weight)
            nn.init.constant(0.01)(self.actor_mean.bias)
            nn.init.he_normal()(self.actor_log_std.weight)
            nn.init.constant(0.01)(self.actor_log_std.bias)
    
    def __call__(self, x):
        for layer in self.layers:
            x = mx.maximum(layer(x), 0.0) 
        
        if self.method_type == 'actor_critic':
            return self.actor_head(x), self.critic_head(x)
        elif self.method_type == 'soft_ac':
            mean = self.actor_mean(x)
            log_std = self.actor_log_std(x)
            return mean, log_std
        else:  # Q-learning or Policy Gradient
            return self.layers[-1](x) 