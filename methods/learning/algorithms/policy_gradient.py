import mlx.core as mx
import mlx.nn as nn

def compute_returns(rewards, gamma):
    returns = mx.zeros_like(rewards)
    running_return = 0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns

def policy_loss(policy_network, states, actions, returns):
    logits = policy_network(states)
    # Handle different shapes based on network type
    if logits.ndim == 3:  # RNN output: (batch, sequence, actions)
        logits = logits[:, -1, :]  # Take the last sequence step
    action_probs = nn.softmax(logits)
    selected_action_probs = mx.take_along_axis(action_probs, mx.expand_dims(actions, axis=1), axis=1)
    log_probs = mx.log(selected_action_probs)
    return -mx.mean(log_probs * returns)

def update_policy_network(policy_network, optimizer, states, actions, rewards, gamma):
    returns = compute_returns(rewards, gamma)
    loss_and_grad_fn = nn.value_and_grad(policy_network, lambda net: policy_loss(net, states, actions, returns))
    loss, grads = loss_and_grad_fn(policy_network)
    optimizer.update(policy_network, grads)
    return loss.item()

def select_action(policy_network, state):
    state_input = mx.expand_dims(state, axis=0)
    if len(state.shape) == 2:  # RNN input
        state_input = mx.expand_dims(state_input, axis=0)  # (1, sequence_n, n_receptors)
    logits = policy_network(state_input)
    if logits.ndim == 3:  # RNN output
        logits = logits[:, -1, :]  # Take the last sequence step
    probs = nn.softmax(logits[0])
    return mx.random.categorical(probs).item()