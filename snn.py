import random
import numpy as np
import time

class Neuron:
    def __init__(self, id, neuron_type):
        self.id = id
        self.neuron_type = neuron_type
        self.spiked = False
        
        # STDP
        self.last_sent = None
        self.last_received = {}
        self.ltp_rate = 0.01
        self.ltd_rate = 0.001
        
        # WEIGHTS
        self.weights = {}
        self.connections = []
        
        # MEMBRANE POTENTIAL
        self.membrane_potential = 0
        self.m_decay = 0.95
        self.threshold = 1
        self.reset = 0

        # LEARNING 
        self.learning_rate = 0.001

        # ELIGIBILITY
        self.eligibility = {}
        self.e_decay = 0.99

        # REFRACTORY
        self.refractory_period = 5
        self.refractory_counter = 0

        # ACTIVITY
        self.base_activity = 0
        
        # PAPERWORK
        self.print_counter = 0

    def set_weights(self, weights):
        self.weights = weights
        self.setup_eligibility(weights)

    def setup_eligibility(self, weights):
        # Correct assumption: weights is a dictionary with each key corresponding to a source neuron
        # Initialize eligibility with a scalar value for each connection
        self.eligibility = {source: 0 for source in weights}

    def connect_to(self, other_neuron):
        self.connections.append(other_neuron)

    def receive_spike(self, spike, source):
        # if self.refractory_counter > 0:
        #     # TODO DETERMINE HOW TO HANDLE REFRACTORY PERIOD
        #     self.membrane_potential += spike * self.weights[source] * self.m_decay
        #     self.eligibility[source] += spike * self.weights[source] * self.m_decay
        # else:
        #     self.membrane_potential += spike * self.weights[source]
        #     self.eligibility[source] += spike * self.weights[source]

        if self.refractory_counter == 0:
            self.membrane_potential += spike * self.weights[source]
            self.eligibility[source] += spike * self.weights[source]

        self.last_received[source] = time.time()

    def spike_timing(self, last_sent):
        for key in self.last_received:
            time_difference = last_sent - self.last_received[key]

            if time_difference > 0:
                # LTP: Presynaptic spike before postsynaptic spike
                weight_change = self.learning_rate * np.exp(-abs(time_difference) / self.ltp_rate)
            else:
                # LTD: Postsynaptic spike before presynaptic spike
                weight_change = -self.learning_rate * np.exp(-abs(time_difference) / self.ltd_rate)

            self.weights[key] += weight_change

    def send_spike(self):
        if self.neuron_type == 'output':
            self.spiked = True
            # print(f'Neuron {self.id} spiked!')
        else:
            for i in range(len(self.connections)):
                self.connections[i].receive_spike(1, self.id)
                self.reset_membrane_potential()
                # print(f'Neuron {self.id} spiked!')

        # STDP
        self.last_sent = time.time()
        self.spike_timing(self.last_sent)

    def input_spike(self, spike):
        self.membrane_potential += spike

    def reset_membrane_potential(self):
        self.membrane_potential = self.reset

    def intrinsic_activity(self):
        if self.refractory_counter == 0:
            print(self.base_activity)
            if np.random.uniform(0, 1) < self.base_activity / 100:
                self.membrane_potential += np.random.normal(0.1, 1)

    def update_weights(self, global_signal):
        # Assuming global_signal is a scalar and self.eligibility is properly managed as a dictionary of scalar values
        for source in self.weights:
            # Ensure self.eligibility[source] is treated as a scalar
            # print(f'Eligibity for {source}: {self.eligibility[source]}')
            # print(f'Global Signal: {global_signal}')
            
            adjusted_trace = self.eligibility[source] * global_signal  # This should be a scalar operation

            # Compute the weight update, which is also a scalar operation
            weight_update = self.learning_rate * adjusted_trace

            if global_signal < 0:
                weight_update = weight_update / 2

            # if self.print_counter > 5000:
            #     print(f'Eligibility: {self.eligibility[source]}')
            #     print(f'Global Signal: {global_signal}')
            #     print(f'Adjusted Trace: {adjusted_trace}')
            #     print(f'Weight Update: {weight_update}')
            #     self.print_counter = 0

            # Apply the weight update
            if global_signal != 0:
                self.weights[source] += weight_update
            
            # self.print_counter += 1
            
    def update(self, global_signal, absence_counter):
        if self.membrane_potential > self.threshold:
            self.send_spike()
            self.membrane_potential = self.reset
            self.refractory_counter = self.refractory_period

        self.membrane_potential *= self.m_decay
        
        # Update eligibility traces for each connection
        for key in self.eligibility:
            # Directly apply decay or any other necessary update to the eligibility trace
            self.eligibility[key] *= self.e_decay

        if self.refractory_counter > 0:
            self.refractory_counter -= 1


        self.base_activity = absence_counter

        self.update_weights(global_signal)

        if self.refractory_counter == 0:
            if self.neuron_type == 'hidden':
                    self.intrinsic_activity()


class Layer:
    def __init__(self, index, size, layer_type):
        self.neurons = []
        self.index = index
        self.layer_type = layer_type

        for i in range(size):
            neuron_type = self.determine_neuron_type()  # Determine the neuron type based on layer type
            self.neurons.append(Neuron(f'l{index}-n{i}', neuron_type))

    def determine_neuron_type(self):
        if self.layer_type == 'input':
            return 'input'
        elif self.layer_type == 'output':
            return 'output'
        else:
            return 'hidden'

    def connect_to(self, next_layer):
        for i in range(len(self.neurons)):
            for j in range(len(next_layer.neurons)):
                self.neurons[i].connect_to(next_layer.neurons[j])

    def initialise_weights(self, network):
        if self.index == 1:  # Skip the first layer as it has no previous layer
            return

        prev_layer = network.get_layer(self.index - 1)
        for neuron in self.neurons:
            weights = {}
            # Iterate over neurons in the previous layer to set weights
            for prev_neuron in prev_layer.neurons:
                # Set the weight for the connection from the previous neuron
                weights[prev_neuron.id] = random.random()
            neuron.set_weights(weights)

        


class Network:
    def __init__(self, architecture, depth, parameters):
        self.layers = []
        # Store references to the layers by their index for easy access
        self.layer_dict = {}

        self.global_signal = 0
        self.signal_decay = 0.6
        self.absence_decay = 0.95

        self.frequency_counter = 1
        self.absence_counter = 0

        self.print_counter = 0

        self.learning_rate = parameters[0]
        self.eligibility_decay = parameters[1]

        # Determine the types of layers based on their position in the network
        layer_types = ['input'] + ['hidden'] * (depth - 1) + ['output']

        # Create layers based on the provided dimensions and depth
        for i, size in enumerate(architecture):
            layer_type = layer_types[min(i, len(layer_types) - 1)]
            layer = Layer(i, size, layer_type)  # Use i instead of i + 1
            self.layers.append(layer)
            self.layer_dict[layer.index] = layer

        # Connect all layers from the first to the last
        for i in range(len(self.layers) - 1):
            self.layers[i].connect_to(self.layers[i + 1])






    def construct(self, weights, parameters):
        # Connect all layers from the first to the last
        for i in range(len(self.layers) - 1):  # Adjusted to iterate from the first layer
            self.layers[i].connect_to(self.layers[i + 1])  # Adjusted to connect forward
        
        self.learning_rate = parameters[0]
        self.eligibility_decay = parameters[1]

        self.set_weights(weights, self.learning_rate, self.eligibility_decay)

    def set_weights(self, weights, learning_rate, eligibility_decay):
        # Group weights by target neuron
        grouped_weights = {}
        for key, value in weights.items():
            source_id, target_id = key.split('_')
            if target_id not in grouped_weights:
                grouped_weights[target_id] = {}
            grouped_weights[target_id][source_id] = value

        # Set weights for each target neuron
        for target_id, weights in grouped_weights.items():
            target_layer_index, target_neuron_index = map(int, target_id[1:].split('-n'))
            target_neuron = self.layer_dict[target_layer_index].neurons[target_neuron_index]
            target_neuron.set_weights(weights)
            # print(target_neuron.weights)
            target_neuron.learning_rate = learning_rate
            target_neuron.eligibility_decay = eligibility_decay
            # print(target_neuron.learning_rate)

    def get_layer(self, index):
        # Retrieve a layer by its index
        return self.layer_dict.get(index)
    
    def propagate_spike(self, spikes, count):
        for i in spikes:
            input_neuron = self.get_layer(0).neurons[i]
            if spikes[i] == 1:
                input_neuron.input_spike(1)
                self.frequency_counter += 1
                self.absence_counter -= 1

                if count <= 9:
                    reward = (0.5 * (count + 1)) * ((self.frequency_counter + 2) / 2)
                else:
                    reward = (0.05 * (count + 10)) * ((self.frequency_counter + 2) / 2)
                self.global_signal += max(reward, 0)
            else:
                input_neuron = self.get_layer(0).neurons[i+2]
                input_neuron.input_spike(1)
                self.absence_counter += 1
                self.global_signal -= self.absence_counter / 1000
                # print(self.absence_counter)
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.update(self.global_signal, self.absence_counter)
        output = self.get_outputs()

        if self.frequency_counter > 0.5:
            self.frequency_counter *= self.signal_decay

        self.global_signal *= self.signal_decay
        self.absence_counter *= self.absence_decay

        return output

    def get_outputs(self):
        # Retrieve the outputs from the output layer
        output = [0, 0, 0]
        spiked = False
        last_layer_index = len(self.layers) - 1
        for neuron in self.get_layer(last_layer_index).neurons:
            if neuron.spiked:
                neuron.spiked = False
                output[int(neuron.id[-1])] = 1
                spiked = True
        if spiked == True:
            if self.print_counter > 50:
                print(f'Action potentials: {output}')
                self.print_counter = 0
            self.print_counter += 1

        # ACTIONS BY MEMBRANE POTENTIAL INSTEAD OF SPIKES
        # if spiked == False:
        #     for neuron in self.get_layer(last_layer_index).neurons:
        #         output[int(neuron.id[-1])] = neuron.membrane_potential
        #     if self.print_counter > 50:
        #         print(f'Membrane potentials: {output}')
        #         self.print_counter = 0
        #     self.print_counter += 1
        #     max_potential = max(output)
        #     output = [1 if potential == max_potential else 0 for potential in output]
        return output
        
            
    def modify_learning(self, reward_signal):
        self.global_signal += reward_signal
        # print(self.global_signal)

    def update_signal(self):
        self.global_signal *= self.signal_decay
        # print(self.global_signal)
    
    def reset_membranes(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.membrane_potential = 0
                neuron.refractory_counter = 0


    def retrieve_weights(self):
        weights = {}
        for i in range(1, len(self.layers)):  # Skip the input layer
            layer = self.layers[i]
            for j, neuron in enumerate(layer.neurons):
                for source_id, weight in neuron.weights.items():
                    # Extract the index of the source neuron from its id
                    k = int(source_id.split('-n')[-1])
                    weights[f'l{i-1}-n{k}_l{i}-n{j}'] = weight
        return weights
        
                


