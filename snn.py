import numpy as np
import mlx.core as mx
import threading
import time
from evolve import Individual, Evolution

class Layer:
    def __init__(self, n_in, n_out, weights):
        self.weights = mx.array(weights)
        # self.weights = mx.array(np.random.uniform(-1, 1, (n_in, n_out)))
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

class Network:
    def __init__(self, individual):
        dim = individual.architecture
        weights = Evolution.rehydrate(individual)
        self.layers = [Layer(dim[i], dim[i+1], weights[i]) for i in range(len(dim)-1)]

    def forward(self, spikes):
        for layer in self.layers:
            spikes = layer.forward(spikes)
        return spikes