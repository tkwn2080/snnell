import mlx.core as mx

# mx.matmul for matrix multiplication: spike * weights
# n input neurons, m connected neurons (layer one)
# if dense connections, then connections = n * m
# weights = connections (n * m)
# mx.core.sum for summing the spikes to a neuron
# mx.core.greater_equal for checking the threshold

# store the weights in a matrix
# store the spikes in a matrix
# membrane_potentials = mx.core.sum(mx.core.matmul(spikes, weights))
# store the membrane potentials in a matrix
# store the thresholds in a matrix
# spikes = mx.core.greater_equal(membrane_potentials, thresholds)
# repeat for each layer

architecture = [2,3]

spikes1 = mx.array([1,0])
# spikes1 = mx.broadcast_to(spikes1, (3,2))

weights1_2 = mx.array([1,0.9,0.8,0.7,0.6,0.5])
print(f'weights1_2: {weights1_2}')
weights1_2 = mx.reshape(weights1_2, (2,3))
print(f'weights1_2: {weights1_2}')


membrane_potentials2 = mx.array([0,0,0])
thresholds2 = mx.array([1,1,1])
spikes2 = mx.array([0,0,0])
print(f'membrane_potentials2: {membrane_potentials2}')

membrane_potentials2 = mx.matmul(spikes1, weights1_2)
print(f'membrane_potentials2: {membrane_potentials2}')

thresholds = [1,1,1]
spikes2 = mx.greater_equal(membrane_potentials2, thresholds2)
print(f'spikes2: {spikes2}')


mask = mx.where(spikes2, mx.zeros_like(spikes2), membrane_potentials2)
print(f'mask: {mask}')



def forward(spikes, weights, dimensions, membrane_potentials, thresholds, refractory_periods):
    # This takes the spikes from a layer (with weights and thresholds)
    # It returns the spikes for the next layer

    refractory_period = 10
    decay_rate = 0.5

    # Spikes are multiplied by their corresponding weights
    weights = mx.reshape(weights, dimensions)
    input_potentials = mx.matmul(spikes, weights)

    # If the refractory period >= 1, then the effect is nullified
    # Simultaneously reduces the refractory period by 1
    refractory, refractory_periods = refractory_counter(refractory_periods)
    input_potentials = mx.multiply(input_potentials, refractory)

    # Input potentials are added to the extant potential
    membrane_potentials = mx.add(membrane_potentials, input_potentials)

    # If the membrane potential >= threshold, sends a spike
    spikes = mx.greater_equal(membrane_potentials, thresholds)
    spikes_out = mx.where(spikes, mx.ones(spikes.shape, dtype=mx.int32), mx.zeros(spikes.shape, dtype=mx.int32))
    
    # If the neuron spikes, then reset + initialise refractory period
    membrane_potentials = reset(membrane_potentials, spikes)

    # Decay the membrane potentials for this timestep
    membrane_potentials = decay(membrane_potentials, decay_rate)
    
    return membrane_potentials, spikes_out, refractory_periods

def reset(membrane_potentials, spikes):
    # This takes the membrane potentials and spikes from within a layer
    # It returns the membrane potentials for that layer, based on the spikes
    
    membrane_potentials = mx.where(spikes, mx.zeros_like(membrane_potentials), membrane_potentials)
    
    return membrane_potentials

def refractory_counter(refractory_periods):
    # This takes the refractory periods from within a layer
    # It returns a mask based on the refractory periods

    # If refractory period >= 1, then returns a zero-mask
    refractory = mx.greater_equal(refractory_periods, mx.ones(refractory_periods.shape, dtype=mx.int32))
    refractory = mx.where(refractory, mx.zeros(refractory.shape, dtype=mx.int32), mx.ones(refractory.shape, dtype=mx.int32))

    # Inverts the refractory mask, subtracts from the refractory periods
    # TODO Surely there must be a better way to do this?
    refractory_reduce = mx.where(refractory, mx.zeros(refractory.shape, dtype=mx.int32), mx.ones(refractory.shape, dtype=mx.int32))
    refractory_periods = mx.subtract(refractory_periods, refractory_reduce)

    return refractory, refractory_periods

for i in range(1000):
    test = forward(mx.array([1,1]), mx.array([1,0.9,0.8,0.7,0.6,0.5]), [2,3], mx.array([0,0,0]), mx.array([1,1,1]), mx.array([0,0,999]))
    if i == 999:
        print(f'test: {test}')
    else:
        print(i)


def decay(membrane_potentials, decay_rate):
    # This takes the membrane potentials for a timestep
    # It returns the membrane potentials for the next timestep

    output = mx.multiply(membrane_potentials, decay_rate)

    return output

# Each layer should have a set of arrays
# Weights, with dimensions corresponding to the number of neurons in the layer
# For a layer with n neurons and m connections, the weights should be a matrix of size n * m
# Spikes, of a length corresponding to the number of neurons in the layer
# Membrane potentials, of a length corresponding to the number of neurons in the layer

# This presumably means doing more at the layer level than the neuron level

