import mlx.core as mx
import mlx.nn as nn

def q_loss(q_network, state, action, reward, next_state, done, gamma):
    current_q = q_network(state)[0, action]
    next_q = mx.max(q_network(next_state), axis=1)
    target_q = reward + (1 - float(done)) * gamma * next_q
    return mx.mean((current_q - target_q) ** 2)

def update_q_network(q_network, optimizer, state, action, reward, next_state, done, gamma):
    loss_and_grad_fn = nn.value_and_grad(q_network, lambda net: q_loss(net, state, action, reward, next_state, done, gamma))
    loss, grads = loss_and_grad_fn(q_network)
    
    optimizer.update(q_network, grads)
    
    return loss.item()

def epsilon_greedy_action(q_network, state, epsilon, n_actions):
    if mx.random.uniform() < epsilon:
        action = mx.random.randint(0, n_actions, (1,)).item()
        return action
    else:
        q_values = q_network(state)
        # Ensure we're always dealing with a 2D array (batch, actions)
        if q_values.ndim == 3:  # (batch, sequence, actions)
            q_values = q_values[:, -1, :]  # Take the last sequence step
        elif q_values.ndim == 1:  # (actions,)
            q_values = mx.expand_dims(q_values, axis=0)  # Add batch dimension
        
        # Now q_values should always be (batch, actions)
        actions = mx.argmax(q_values, axis=-1)
        return actions[0].item()  # Return the action for the first (and only) batch item