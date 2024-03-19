import numpy as np
import mlx.core as mx
import threading
import time
from evolve import Individual, Evolution

class Layer:
    def __init__(self, n_in, n_out, weights):
        self.weights = mx.array(weights)
        self.mem_p = mx.zeros((n_out,), dtype=mx.float32)
        self.e_trace = mx.zeros((n_out,), dtype=mx.float32)
        self.thresholds = mx.ones((n_out,), dtype=mx.float32)
        self.decay_rate = 0.8

    def forward(self, spikes_in):
        # Calculate weighted input from spikes
        weighted_input = mx.matmul(spikes_in, self.weights)

        # Add weighted input to membrane potentials
        self.mem_p = mx.add(self.mem_p, weighted_input)

        # Add weighted input to eligibility trace
        self.e_trace = mx.add(self.e_trace, weighted_input)

        # Check if membrane potential is greater than threshold
        spikes_out = mx.greater_equal(self.mem_p, self.thresholds)

        # Reset membrane potentials to zero for spikes
        self.mem_p = mx.where(spikes_out, mx.zeros_like(self.mem_p), self.mem_p)

        # Update membrane potentials based on decay rate
        self.mem_p = mx.multiply(self.mem_p, self.decay_rate)

        # Update eligibility trace based on decay rate
        self.e_trace = mx.multiply(self.e_trace, self.decay_rate)

        return spikes_out

class IzhikevichLayer(Layer):
    def __init__(self, num_neurons, weights, a=0.02, b=0.2, c=-65, d=8):
        super().__init__(weights.shape[0], weights.shape[1], weights)  # Call the parent class constructor
        self.v = mx.ones((num_neurons,), dtype=mx.float32) * c  # Membrane potentials
        self.u = mx.zeros((num_neurons,), dtype=mx.float32)  # Recovery variables
        self.a = a  # Izhikevich model parameter a
        self.b = b  # Izhikevich model parameter b
        self.c = c  # Izhikevich model parameter c
        self.d = d  # Izhikevich model parameter d
        self.threshold = 30  # Spiking threshold

    def add_current(self, input_currents):
        self.v += input_currents

    def forward(self, spikes_in):
        # Calculate weighted input from spikes
        weighted_input = mx.matmul(spikes_in, self.weights)

        # Update membrane potentials and recovery variables based on Izhikevich model equations
        self.v = mx.add(mx.add(mx.add(mx.multiply(0.04, mx.square(self.v)), mx.multiply(5, self.v)), 140), mx.subtract(weighted_input, self.u))
        self.u = mx.add(self.u, mx.multiply(self.a, mx.subtract(mx.multiply(self.b, self.v), self.u)))

        # Check if membrane potentials exceed the threshold
        spikes_out = mx.greater_equal(self.v, self.threshold)

        # Reset membrane potentials and update recovery variables for spiking neurons
        self.v = mx.where(spikes_out, self.c, self.v)
        self.u = mx.where(spikes_out, mx.add(self.u, self.d), self.u)

        return spikes_out

    def output(self, spikes_in):
        # Calculate weighted input from spikes
        weighted_input = mx.matmul(spikes_in, self.weights)

        # Update membrane potentials and recovery variables based on Izhikevich model equations
        self.v = mx.add(mx.add(mx.add(mx.multiply(0.04, mx.square(self.v)), mx.multiply(5, self.v)), 140), mx.subtract(weighted_input, self.u))
        self.u = mx.add(self.u, mx.multiply(self.a, mx.subtract(mx.multiply(self.b, self.v), self.u)))

        # Check if membrane potentials exceed the threshold
        spikes_out = mx.greater_equal(self.v, self.threshold)

        # Reset membrane potentials and update recovery variables for spiking neurons
        self.v = mx.where(spikes_out, self.c, self.v)
        self.u = mx.where(spikes_out, mx.add(self.u, self.d), self.u)

        return self.v

class IzhikevichInputLayer(IzhikevichLayer):
    def __init__(self, num_neurons, a=0.02, b=0.2, c=-65, d=8):
        super().__init__(num_neurons, mx.eye(num_neurons), a, b, c, d)  # Identity matrix as weights

    def forward(self, input_currents):
        # Update membrane potentials and recovery variables based on Izhikevich model equations
        self.v = mx.add(mx.add(mx.add(mx.multiply(0.04, mx.square(self.v)), mx.multiply(5, self.v)), 140), mx.subtract(input_currents, self.u))
        self.u = mx.add(self.u, mx.multiply(self.a, mx.subtract(mx.multiply(self.b, self.v), self.u)))

        # Check if membrane potentials exceed the threshold
        spikes_out = mx.greater_equal(self.v, self.threshold)

        # Reset membrane potentials and update recovery variables for spiking neurons
        self.v = mx.where(spikes_out, self.c, self.v)
        self.u = mx.where(spikes_out, mx.add(self.u, self.d), self.u)

        return spikes_out

class Network:
    def __init__(self, individual, type):
        dim = individual.architecture
        weights = individual.rehydrate()

        if type == 'izhikevich':
            self.layers = [IzhikevichInputLayer(dim[0])]  # Input layer
            self.layers.extend([IzhikevichLayer(dim[i+1], weights[i]) for i in range(len(dim)-1)])  # Hidden and output layers
        elif type == 'standard':
            self.layers = [Layer(dim[i], dim[i+1], weights[i]) for i in range(len(dim)-1)]

    def inject_current(self, input_currents):
        # Pass the input currents directly to the input layer
        input_layer = self.layers[0]
        input_layer.add_current(input_currents)
        
        # Call the forward method to propagate the spikes through the network
        output_spikes = self.forward(mx.zeros((input_layer.v.shape[0],), dtype=mx.float32), 'mem_p')
        
        return output_spikes

    def forward(self, spikes, type):
        if type == 'spikes':
            for layer in self.layers:
                spikes = layer.forward(spikes)
            return spikes
        elif type == 'mem_p':
            for layer in self.layers[:-1]:
                spikes = layer.forward(spikes)
            output = self.layers[-1].output(spikes)
            return output
        

# Implement STDP, LTP, LTD
# Implement e-prop + global signal
# Implement recurrent connections
# Implement actor-critic architecture