import mlx.core as mx
import mlx.nn as nn

class RecurrentNetwork(nn.Module):
    def __init__(self, n_receptors, n_hidden, n_layers, n_out, sequence_n):
        super().__init__()
        self.n_receptors = n_receptors
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_out = n_out
        self.sequence_n = sequence_n
        
        self.rnn_layers = [nn.RNN(n_receptors if i == 0 else n_hidden, n_hidden) for i in range(n_layers)]
        
        if isinstance(n_out, tuple):  # Actor-Critic
            self.actor_head = nn.Linear(n_hidden, n_out[0])
            self.critic_head = nn.Linear(n_hidden, n_out[1])
        else:  # Q-learning or Policy Gradient
            self.output_layer = nn.Linear(n_hidden, n_out)

    def __call__(self, x):
        # x should have shape (batch_size, sequence_n, n_receptors)
        for rnn in self.rnn_layers:
            x = rnn(x)
        
        # Take the output of the last time step
        last_hidden = x[:, -1, :]
        
        # Pass through the output layer(s)
        if hasattr(self, 'actor_head'):  # Actor-Critic
            return self.actor_head(last_hidden), self.critic_head(last_hidden)
        else:  # Q-learning or Policy Gradient
            return self.output_layer(last_hidden)