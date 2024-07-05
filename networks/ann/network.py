import mlx.core as mx
import mlx.nn as nn

class Network(nn.Module):
    def __init__(self, n_in, n_hidden, n_layers, n_out):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        
        # Input layer
        self.layers = [nn.Linear(n_in, n_hidden)]
        
        # Hidden layers
        self.layers.extend([nn.Linear(n_hidden, n_hidden) for _ in range(n_layers - 1)])
        
        # Output layer(s)
        if isinstance(n_out, tuple):  # Actor-Critic
            self.actor_head = nn.Linear(n_hidden, n_out[0])
            self.critic_head = nn.Linear(n_hidden, n_out[1])
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
    
    def __call__(self, x):
        for layer in self.layers:
            x = mx.maximum(layer(x), 0.0)  # ReLU activation
        
        if hasattr(self, 'actor_head'):  # Actor-Critic
            return self.actor_head(x), self.critic_head(x)
        else:  # Q-learning or Policy Gradient
            return x  # The last layer output is already linear