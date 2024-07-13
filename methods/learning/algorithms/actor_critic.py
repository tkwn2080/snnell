import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

def compute_returns(rewards, values, gamma):
    returns = mx.zeros_like(rewards)
    advantages = mx.zeros_like(rewards)
    last_return = 0
    for t in reversed(range(len(rewards))):
        returns[t] = rewards[t] + gamma * last_return
        advantages[t] = returns[t] - values[t]
        last_return = returns[t]
    return returns, advantages

def actor_critic_loss(actor_critic_network, states, actions, returns, advantages):
    logits, values = actor_critic_network(states)
    
    # Handle different shapes based on network type
    if logits.ndim == 3:  # RNN output: (batch, sequence, actions)
        logits = logits[:, -1, :]  # Take the last sequence step
        values = values[:, -1, :]

    # Actor loss
    action_probs = nn.softmax(logits)
    selected_action_probs = mx.take_along_axis(action_probs, mx.expand_dims(actions, axis=1), axis=1)
    log_probs = mx.log(selected_action_probs)
    actor_loss = -mx.mean(log_probs * advantages)

    # Critic loss
    critic_loss = mx.mean((returns - values.squeeze()) ** 2)

    # Entropy for exploration
    entropy = -mx.sum(action_probs * mx.log(action_probs + 1e-10), axis=1)
    entropy_loss = -mx.mean(entropy)

    # Combine losses
    total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

    return total_loss, actor_loss, critic_loss, entropy_loss

def update_actor_critic_network(actor_critic_network, optimizer, states, actions, rewards, gamma, max_grad_norm):
    _, values = actor_critic_network(states)
    if values.ndim == 3:  # RNN output
        values = values[:, -1, :]
    returns, advantages = compute_returns(rewards, values.squeeze(), gamma)
    
    def loss_fn(network):
        return actor_critic_loss(network, states, actions, returns, advantages)[0]
    
    loss_and_grad_fn = nn.value_and_grad(actor_critic_network, loss_fn)
    (total_loss), grads = loss_and_grad_fn(actor_critic_network)
    
    # Compute individual losses for logging
    _, actor_loss, critic_loss, entropy_loss = actor_critic_loss(actor_critic_network, states, actions, returns, advantages)
    
    # Gradient clipping
    clipped_grads, _ = optim.clip_grad_norm(grads, max_grad_norm)
    
    optimizer.update(actor_critic_network, clipped_grads)
    return total_loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()

def select_action(actor_critic_network, state):
    state_input = mx.expand_dims(state, axis=0)
    if len(state.shape) == 2:  # RNN input
        state_input = mx.expand_dims(state_input, axis=0)  # (1, sequence_n, n_receptors)
    logits, _ = actor_critic_network(state_input)
    if logits.ndim == 3:  # RNN output
        logits = logits[:, -1, :]  # Take the last sequence step
    probs = nn.softmax(logits[0])
    return mx.random.categorical(probs).item()