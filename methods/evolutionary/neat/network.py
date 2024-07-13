import numpy as np
from scipy.sparse import csr_matrix

class SNNSimulator:
    def __init__(self, node_array, weight_matrix, null_mask):
        self.network_type = 'spiking'
        self.node_array = node_array
        self.weight_matrix = weight_matrix
        self.null_mask = null_mask
        self.num_nodes = len(node_array)
        self.num_inputs = np.sum(node_array['node_type'] == 'input')
        self.num_outputs = np.sum(node_array['node_type'] == 'output')
        self.v = np.full(self.num_nodes, -65.0)  # Initial membrane potential
        self.u = np.zeros(self.num_nodes)  # Initial recovery variable
        self.last_output = np.zeros(self.num_nodes)  # Last output state

    def reset(self):
        self.v = np.full(self.num_nodes, -65.0)
        self.u = np.zeros(self.num_nodes)
        self.last_output = np.zeros(self.num_nodes)

    def simulate(self, input_current, steps):
        output = np.zeros((steps, self.num_nodes))
        for t in range(steps):
            # Apply input current
            I = np.zeros(self.num_nodes)
            I[:self.num_inputs] = input_current[t]  # Apply to all input nodes

            # Scale weights to allow inhibitory connections (e.g., -5 to 5)
            scaled_weights = 60 * self.weight_matrix

            # Compute synaptic input
            synaptic_input = scaled_weights.dot(self.last_output)

            # Update membrane potential and recovery variable
            self.v += 0.5 * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I + synaptic_input)
            self.v += 0.5 * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I + synaptic_input)
            self.u += self.node_array['a'] * (self.node_array['b'] * self.v - self.u)

            # Check for spikes
            fired = self.v >= 30
            output[t, fired] = 1
            self.v[fired] = self.node_array['c'][fired]
            self.u[fired] += self.node_array['d'][fired]

            # Update last_output for the next iteration
            self.last_output = output[t]

        return output

    def propagate(self, input_current, input_duration, propagation_steps):
        if len(input_current) != self.num_inputs:
            raise ValueError(f"Input current length {len(input_current)} does not match number of input nodes {self.num_inputs}")

        # Prepare input by repeating for input_duration
        prepared_input = np.repeat(input_current.reshape(1, -1), input_duration, axis=0)
        
        # Pad with zeros if propagation_steps > input_duration
        if propagation_steps > input_duration:
            padding = np.zeros((propagation_steps - input_duration, self.num_inputs))
            prepared_input = np.vstack((prepared_input, padding))
        
        # Simulate the network
        all_output = self.simulate(prepared_input, propagation_steps)
        
        # Return only the output node activities
        return all_output[:, -self.num_outputs:]