import numpy as np
import mlx.core as mx
import threading
import time
from evolve import Individual, Evolution

class Layer:
    # N.B. I suspect this will not work, largely deprecated for Izhikevich model
    def __init__(self, n_in, n_out, weights, recurrent_weights=None):
        self.weights = mx.array(weights)
        self.recurrent_weights = mx.array(recurrent_weights) if recurrent_weights is not None else None
        self.mem_p = mx.zeros((n_out,), dtype=mx.float32)
        self.e_trace = mx.zeros((n_out,), dtype=mx.float32)
        self.thresholds = mx.ones((n_out,), dtype=mx.float32)
        self.decay_rate = 0.8

    def forward(self, spikes_in, recurrent_output=None):
        # Calculate weighted input from spikes
        weighted_input = mx.matmul(spikes_in, self.weights)

        # Add recurrent input if available
        if self.recurrent_weights is not None and recurrent_output is not None:
            recurrent_input = mx.matmul(recurrent_output, self.recurrent_weights)
            weighted_input = mx.add(weighted_input, recurrent_input)

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
    def __init__(self, num_neurons, weights, recurrent_weights=None, a=0.02, b=0.2, c=-65, d=8):
        super().__init__(weights.shape[0], weights.shape[1], weights, recurrent_weights)
        self.v = mx.ones((num_neurons,), dtype=mx.float32) * c
        self.u = mx.zeros((num_neurons,), dtype=mx.float32)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.threshold = 30
        

    def forward(self, spikes_in, recurrent_output=None):
        # Calculate weighted input from spikes
        weighted_input = mx.matmul(spikes_in, self.weights)

        # Add recurrent input if available
        if self.recurrent_weights is not None and recurrent_output is not None:
            recurrent_output = mx.reshape(recurrent_output, (-1, 1))  # Reshape recurrent output to match expected shape
            recurrent_input = mx.matmul(self.recurrent_weights, recurrent_output)
            weighted_input = mx.add(weighted_input, mx.reshape(recurrent_input, (-1,)))

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

        # Store the membrane potentials before resetting
        output_potentials = self.v

        # Reset membrane potentials and update recovery variables for spiking neurons
        self.v = mx.where(spikes_out, self.c, self.v)
        self.u = mx.where(spikes_out, mx.add(self.u, self.d), self.u)

        return output_potentials

class IzhikevichInputLayer(IzhikevichLayer):
    def __init__(self, num_neurons):
        super().__init__(num_neurons, mx.eye(num_neurons))  # Identity matrix as weights
        
        num_antennae = 6
        num_cilia = num_neurons - num_antennae
        
        a_cilia, b_cilia, c_cilia, d_cilia = 0.02, 0.2, -50, 2
        a_antennae, b_antennae, c_antennae, d_antennae = 0.02, 0.2, -65, 8

        # Initialize the parameter arrays
        self.a = mx.zeros(num_neurons, dtype=mx.float32)
        self.b = mx.zeros(num_neurons, dtype=mx.float32)
        self.c = mx.zeros(num_neurons, dtype=mx.float32)
        self.d = mx.zeros(num_neurons, dtype=mx.float32)
        
        # Initialize Izhikevich parameters for antennae neurons (ordinary spiking)
        self.a[:num_antennae] = a_antennae
        self.b[:num_antennae] = b_antennae
        self.c[:num_antennae] = c_antennae
        self.d[:num_antennae] = d_antennae
        
        # Initialize Izhikevich parameters for cilia neurons (bursting)
        self.a[num_antennae:] = a_cilia
        self.b[num_antennae:] = b_cilia
        self.c[num_antennae:] = c_cilia
        self.d[num_antennae:] = d_cilia
    
    def add_current(self, input_currents):
        self.v += input_currents

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
    def __init__(self, individual, type, recurrent=False):
        dim = individual.architecture
        weights, recurrent_weights = individual.rehydrate()

        self.recurrent_outputs = [None] * (len(dim) - 1)  # Store recurrent outputs for each layer

        if type == 'izhikevich':
            self.layers = [IzhikevichInputLayer(dim[0])]
            recurrent_index = 0  # Initialize a separate index for recurrent weights
            for i in range(1, len(dim)):
                if recurrent and i > 0 and i < len(dim) - 1:
                    self.layers.append(IzhikevichLayer(dim[i], weights[i-1], recurrent_weights[recurrent_index]))
                    recurrent_index += 1  # Increment the recurrent index
                else:
                    self.layers.append(IzhikevichLayer(dim[i], weights[i-1]))
        elif type == 'standard':
            self.layers = [Layer(dim[0], dim[1], weights[0])]
            for i in range(1, len(dim)-1):
                if recurrent and i > 1 and i < len(dim) - 2:
                    self.layers.append(Layer(dim[i], dim[i+1], weights[i], recurrent_weights[i-1]))
                else:
                    self.layers.append(Layer(dim[i], dim[i+1], weights[i]))

    def forward(self, spikes, output_type):
        for i, layer in enumerate(self.layers[:-1]):  # Iterate over all layers except the output layer
            if i == 0:  # Input layer
                spikes = layer.forward(spikes)
            else:  # Hidden layers
                spikes = layer.forward(spikes, self.recurrent_outputs[i-1])
            if i < len(self.layers) - 2:  # Store recurrent output for the next hidden layer
                self.recurrent_outputs[i] = spikes
        
        output_layer = self.layers[-1]  # Get the output layer
        if output_type == 'spikes':
            output = output_layer.forward(spikes)  # Apply forward method to the output layer
            return output
        elif output_type == 'mem_p':
            output = output_layer.output(spikes)  # Apply output method to the output layer
            return output

    def inject_current(self, input_currents):
        # Set number of timesteps
        num_timesteps = 4

        # Set output type: spikes or mem_p
        output_type = 'spikes'
        
        # Initialize output spike buffer
        output_spike_buffer = []
        
        # Pass the input currents directly to the input layer
        input_layer = self.layers[0]
        
        for _ in range(num_timesteps):
            input_layer.add_current(input_currents)
            
            # Call the forward method to propagate the spikes through the network
            output_spikes = self.forward(mx.zeros((input_layer.v.shape[0],), dtype=mx.float32), output_type)
            
            # Accumulate output spikes in the buffer
            output_spike_buffer.append(output_spikes)
        
        # Process the output spike buffer
        output_spikes = self.process_output_spikes(output_spike_buffer)
        
        return output_spikes

    def process_output_spikes(self, output_spike_buffer):
        # A: Set output spikes to a maximum of one
        # output_spikes = mx.max(mx.stack(output_spike_buffer), axis=0)
        
        # B:Accumulate output spikes as discrete integers
        output_spikes = mx.sum(mx.stack(output_spike_buffer), axis=0)
        # print(f'Summed output spikes: {output_spikes}')
        
        return output_spikes
        

# Implement STDP, LTP, LTD
# Implement e-prop + global signal
# Implement actor-critic architecture