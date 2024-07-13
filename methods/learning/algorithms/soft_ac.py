import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def __call__(self, state, action):
        sa = mx.concatenate([state, action], axis=-1)
        return self.q1(sa), self.q2(sa)

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, sequence_n):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_n = sequence_n
        
        self.states = mx.zeros((capacity, sequence_n, state_dim))
        self.actions = mx.zeros((capacity, action_dim))
        self.rewards = mx.zeros((capacity, 1))
        self.next_states = mx.zeros((capacity, sequence_n, state_dim))
        self.dones = mx.zeros((capacity, 1))
        
        self.pointer = 0
        self.size = 0

    def push(self, state_sequence, action, reward, next_state_sequence, done):
        self.states[self.pointer] = mx.array(state_sequence)
        self.actions[self.pointer] = mx.array(action)
        self.rewards[self.pointer] = mx.array([reward])
        self.next_states[self.pointer] = mx.array(next_state_sequence)
        self.dones[self.pointer] = mx.array([float(done)])
        
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        indices = mx.array(indices)
        
        return (
            mx.take(self.states, indices, axis=0),
            mx.take(self.actions, indices, axis=0),
            mx.take(self.rewards, indices, axis=0),
            mx.take(self.next_states, indices, axis=0),
            mx.take(self.dones, indices, axis=0)
        )

    def __len__(self):
        return self.size

def soft_ac_loss(actor_forward, critic_forward, states, actions, rewards, next_states, dones, gamma, alpha):
    actor_output = actor_forward(states)
    last_actor_output = actor_output[:, -1, :] if actor_output.ndim == 3 else actor_output
    mean, log_std = mx.split(last_actor_output, indices_or_sections=2, axis=-1)
    std = mx.exp(log_std)
    normal = mx.random.normal(mean.shape)
    sample = mean + std * normal
    log_prob = -0.5 * ((sample - mean) / std) ** 2 - 0.5 * mx.log(2 * mx.pi) - log_std
    log_prob = mx.sum(log_prob, axis=-1, keepdims=True)

    last_states = states[:, -1, :] if states.ndim == 3 else states
    q1, q2 = critic_forward(last_states, sample)
    min_q = mx.minimum(q1, q2)
    actor_loss = mx.mean(alpha * log_prob - min_q)

    # Compute target Q-values
    next_actor_output = actor_forward(next_states)
    last_next_actor_output = next_actor_output[:, -1, :] if next_actor_output.ndim == 3 else next_actor_output
    next_mean, next_log_std = mx.split(last_next_actor_output, indices_or_sections=2, axis=-1)
    next_std = mx.exp(next_log_std)
    next_normal = mx.random.normal(next_mean.shape)
    next_sample = next_mean + next_std * next_normal
    next_log_prob = -0.5 * ((next_sample - next_mean) / next_std) ** 2 - 0.5 * mx.log(2 * mx.pi) - next_log_std
    next_log_prob = mx.sum(next_log_prob, axis=-1, keepdims=True)

    last_next_states = next_states[:, -1, :] if next_states.ndim == 3 else next_states
    next_q1, next_q2 = critic_forward(last_next_states, next_sample)
    next_min_q = mx.minimum(next_q1, next_q2)
    
    target_q = mx.stop_gradient(rewards + gamma * (1 - dones) * (next_min_q - alpha * next_log_prob))

    current_q1, current_q2 = critic_forward(last_states, actions)
    critic_loss = mx.mean((current_q1 - target_q) ** 2 + (current_q2 - target_q) ** 2)

    return actor_loss, critic_loss

class SoftActorCritic:
    def __init__(self, actor, critic, state_dim, action_dim, buffer_size, batch_size, gamma, alpha, tau, lr, sequence_n, use_fractional_updates=False, update_steps=1):
        self.actor = actor
        self.critic = critic
        self.action_dim = action_dim
        
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim, sequence_n)
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.sequence_n = sequence_n
        self.use_fractional_updates = use_fractional_updates
        self.update_steps = update_steps if use_fractional_updates else 1
        
        self.actor_optimizer = optim.Adam(learning_rate=lr)
        self.critic_optimizer = optim.Adam(learning_rate=lr)
        
        self.state_sequence = []

    def compute_gradients(self, states, actions, rewards, next_states, dones):
        def loss_fn(actor_params, critic_params):
            def actor_forward(x):
                self.actor.update(actor_params)
                return self.actor(x)
            
            def critic_forward(state, action):
                self.critic.update(critic_params)
                return self.critic(state, action)
            
            return soft_ac_loss(actor_forward, critic_forward, states, actions, rewards, next_states, dones, self.gamma, self.alpha)

        actor_params = self.actor.parameters()
        critic_params = self.critic.parameters()
        (actor_loss, critic_loss), (actor_grads, critic_grads) = mx.value_and_grad(loss_fn, argnums=(0, 1))(actor_params, critic_params)
        return actor_grads, critic_grads, actor_loss, critic_loss

    def update(self, state, action, reward, next_state, done):
        self.state_sequence.append(state)
        if len(self.state_sequence) > self.sequence_n:
            self.state_sequence = self.state_sequence[-self.sequence_n:]

        if len(self.state_sequence) < self.sequence_n:
            return None, None

        state_sequence = mx.stack(self.state_sequence)
        next_state_sequence = mx.stack(self.state_sequence[1:] + [next_state])
        
        self.replay_buffer.push(state_sequence, action, reward, next_state_sequence, done)
        
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        actor_grads, critic_grads, actor_loss, critic_loss = self.compute_gradients(states, actions, rewards, next_states, dones)
        
        if self.use_fractional_updates:
            fraction = 1.0 / self.update_steps
            for _ in range(self.update_steps):
                actor_grads_fraction = {k: v * fraction for k, v in actor_grads.items()}
                critic_grads_fraction = {k: v * fraction for k, v in critic_grads.items()}
                
                actor_updates = self.actor_optimizer.apply_gradients(self.actor.parameters(), actor_grads_fraction)
                critic_updates = self.critic_optimizer.apply_gradients(self.critic.parameters(), critic_grads_fraction)
                
                self.actor.update(actor_updates)
                self.critic.update(critic_updates)
        else:
            actor_updates = self.actor_optimizer.apply_gradients(self.actor.parameters(), actor_grads)
            critic_updates = self.critic_optimizer.apply_gradients(self.critic.parameters(), critic_grads)
            
            self.actor.update(actor_updates)
            self.critic.update(critic_updates)

        if done:
            self.state_sequence = []

        return actor_loss.item(), critic_loss.item()

    def select_action(self, state):
        if state.ndim == 3:  # If we receive a full sequence (1, sequence_n, state_dim)
            self.state_sequence = [state[0, i, :] for i in range(state.shape[1])]
        else:  # If we receive a single state (state_dim,)
            self.state_sequence.append(state)
            if len(self.state_sequence) > self.sequence_n:
                self.state_sequence = self.state_sequence[-self.sequence_n:]

        if len(self.state_sequence) < self.sequence_n:
            return mx.zeros(self.action_dim)
        
        state_sequence = mx.stack(self.state_sequence)
        state_sequence = mx.expand_dims(state_sequence, axis=0)
        
        actor_output = self.actor(state_sequence)
        
        # Handle different output shapes
        if actor_output.ndim == 3:
            actor_output = actor_output[:, -1, :]  # Take the last sequence output
        elif actor_output.ndim == 2:
            actor_output = actor_output[0]  # Take the first (and only) row
        
        mean, log_std = mx.split(actor_output, indices_or_sections=2, axis=-1)
        std = mx.exp(log_std)
        normal = mx.random.normal(mean.shape)
        action = mean + std * normal

        return action.squeeze()

    def save(self, path):
        # mx.save(path, self.state_dict())
        pass

    def load(self, path):
        # params = mx.load(path)
        # self.update(params)
        pass