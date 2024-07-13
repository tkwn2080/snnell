import mlx.core as mx
import mlx.nn as nn

class RecurrentNetwork(nn.Module):
    def __init__(self, n_receptors, n_hidden, n_layers, n_out, sequence_n, method_type='q_learning'):
        super().__init__()
        self.n_receptors = n_receptors
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_out = n_out
        self.sequence_n = sequence_n
        self.method_type = method_type

        self.rnn_layers = [nn.RNN(n_receptors if i == 0 else n_hidden, n_hidden) for i in range(n_layers)]

        if method_type == 'actor_critic':
            self.actor_head = nn.Linear(n_hidden, n_out[0])
            self.critic_head = nn.Linear(n_hidden, n_out[1])
        elif method_type == 'soft_ac':
            self.actor_mean_movement = nn.Linear(n_hidden, 1)
            self.actor_log_std_movement = nn.Linear(n_hidden, 1)
            self.actor_mean_rotation = nn.Linear(n_hidden, 1)
            self.actor_log_std_rotation = nn.Linear(n_hidden, 1)
        else:  # Q-learning or Policy Gradient
            self.output_layer = nn.Linear(n_hidden, n_out)

    def __call__(self, x, keep_sequence_dim=False):
        for i, rnn in enumerate(self.rnn_layers):
            x = rnn(x)

        if not keep_sequence_dim:
            x = x[:, -1, :]  # Take only the last timestep if we're not keeping the sequence dimension

        # Pass through the output layer(s)
        if self.method_type == 'actor_critic':
            actor_output = self.actor_head(x)
            critic_output = self.critic_head(x)
            return actor_output, critic_output
        elif self.method_type == 'soft_ac':
            mean_movement = self.actor_mean_movement(x)
            log_std_movement = self.actor_log_std_movement(x)
            mean_rotation = self.actor_mean_rotation(x)
            log_std_rotation = self.actor_log_std_rotation(x)
            movement_output = mx.concatenate([mean_movement, log_std_movement], axis=-1)
            rotation_output = mx.concatenate([mean_rotation, log_std_rotation], axis=-1)
            output = mx.concatenate([movement_output, rotation_output], axis=-1)
            return output
        else:  # Q-learning or Policy Gradient
            output = self.output_layer(x)
            return output