import random
import numpy as np
import time
import math

class Neuron:
    def __init__(self, id, neuron_type):
        self.id = id
        self.neuron_type = neuron_type
        self.spiked = False
        
        # STDP
        self.last_sent = None
        self.last_received = {}
        self.ltp_rate = 0.1
        self.ltd_rate = 0.01
        
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
        self.refractory_period = 10
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
        if self.refractory_counter == 0:
            self.membrane_potential += spike * self.weights[source]
            self.eligibility[source] += spike * self.weights[source]
        elif self.refractory_counter > 0:
            discount_factor = 1 - (self.refractory_counter / 10)
            self.membrane_potential += discount_factor * spike * self.weights[source]
            self.eligibility[source] += discount_factor * spike * self.weights[source] 

        self.last_received[source] = time.time()

    def spike_timing(self, last_sent):
        for key in self.last_received:
            
            if key == self.id:
                continue

            time_difference = last_sent - self.last_received[key]

            if time_difference > 0:
                # LTP: Presynaptic spike before postsynaptic spike
                weight_change = self.learning_rate * np.exp(-abs(time_difference) / self.ltp_rate)
            else:
                # LTD: Postsynaptic spike before presynaptic spike
                weight_change = -self.learning_rate * np.exp(-abs(time_difference) / self.ltd_rate)

            self.weights[key] += weight_change
            # self.weights[key] = np.clip(self.weights[key], -1, 1)  

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
            # print(self.base_activity)
            if np.random.uniform(0, 1) < self.base_activity / 100:
                self.membrane_potential += np.random.normal(0.1, 1)

    def update_weights(self, normalized_eligibility, global_signal):
        # Assuming global_signal is a scalar and normalized_eligibility is properly managed as a dictionary of scalar values
        for source in self.weights:
            # Ensure normalized_eligibility[source] is treated as a scalar
            adjusted_trace = normalized_eligibility[source] * global_signal  # This should be a scalar operation

            # Compute the weight update, which is also a scalar operation
            weight_update = self.learning_rate * adjusted_trace

            # Apply the weight update
            if global_signal != 0:
                self.weights[source] += weight_update * np.sign(self.weights[source])
                self.weights[source] = np.clip(self.weights[source], -1, 1)
            
    def update(self, absence_counter):
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

        # if self.refractory_counter == 0:
        #     if self.neuron_type == 'hidden':
        #             self.intrinsic_activity()


class Layer:
    def __init__(self, index, size, layer_type, layer_variant):
        self.neurons = []
        self.index = index
        self.layer_type = layer_type
        if layer_variant == None:
            self.layer_variant = 'forward'
        else:
            self.layer_variant = layer_variant

        for i in range(size):
            neuron_type = self.determine_neuron_type()  # Determine the neuron type based on layer type
            self.neurons.append(Neuron(f'l{index}-n{i}', neuron_type))

            if self.layer_variant == 'recurrent':
                self.connect_recurrent()
    

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

    def connect_recurrent(self):
        for i in range(len(self.neurons)):
            self.neurons[i].connect_to(self.neurons[i])

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
    def __init__(self, architecture, depth, parameters, recurrence):
        self.layers = []
        # Store references to the layers by their index for easy access
        self.layer_dict = {}

        self.recurrent_layer = recurrence

        self.global_signal = 0
        self.signal_decay = 0.5
        self.absence_decay = 0.95

        self.frequency_counter = 1
        self.absence_counter = 0

        self.print_counter = 0

        self.learning_rate = parameters[0]
        self.eligibility_decay = parameters[1]

        layer_types = ['input'] + ['hidden'] * (depth - 1) + ['output']

        for i, size in enumerate(architecture):
            if self.recurrent_layer == i:
                layer_variant = 'recurrent'
            else:
                layer_variant = None
            layer_type = layer_types[min(i, len(layer_types) - 1)]
            layer = Layer(i, size, layer_type, layer_variant)  # Use i instead of i + 1
            self.layers.append(layer)
            self.layer_dict[layer.index] = layer

        for i in range(len(self.layers) - 1):
            self.layers[i].connect_to(self.layers[i + 1])

    def construct(self, weights, parameters):
        # Connect all layers from the first to the last
        for i in range(len(self.layers) - 1):  
            self.layers[i].connect_to(self.layers[i + 1]) 
        
        self.learning_rate = parameters[0]
        self.eligibility_decay = parameters[1]

        self.set_weights(weights, self.learning_rate, self.eligibility_decay)

    def set_weights(self, weights, learning_rate, eligibility_decay):
        grouped_weights = {}
        for key, value in weights.items():
            if '_rec' in key:
                source_id = target_id = key.split('_rec')[0]
            else:
                source_id, target_id = key.split('_')

            if target_id not in grouped_weights:
                grouped_weights[target_id] = {}
            grouped_weights[target_id][source_id] = value

        for target_id, weights in grouped_weights.items():
            target_layer_index, target_neuron_index = map(int, target_id[1:].split('-n'))
            target_neuron = self.layer_dict[target_layer_index].neurons[target_neuron_index]
            target_neuron.set_weights(weights)
            target_neuron.learning_rate = learning_rate
            target_neuron.eligibility_decay = eligibility_decay

    def set_recurrence(self, weights):
        for key, value in weights.items():
            if '_rec' in key:
                source_id, target_id = key.split('_')[0].split('-')
                source_layer_index, source_neuron_index = map(int, source_id[1:].split('-n'))
                target_layer_index, target_neuron_index = map(int, target_id[1:].split('-n'))
                source_neuron = self.layer_dict[source_layer_index].neurons[source_neuron_index]
                target_neuron = self.layer_dict[target_layer_index].neurons[target_neuron_index]
                source_neuron.connect_to(target_neuron)
                source_neuron.weights[target_id] = value

    def get_layer(self, index):
        # Retrieve a layer by its index
        return self.layer_dict.get(index)
    
    def network_decay(self):
        self.frequency_counter *= self.signal_decay
        self.global_signal *= self.signal_decay
        self.absence_counter *= self.absence_decay
    
    def propagate_spike(self, spikes, count):
        self.network_decay()



        for i, spike in enumerate(spikes):
            if i < 4:
                input_neuron = self.get_layer(0).neurons[i]
                if spike == 1:
                    input_neuron.input_spike(1)
                    self.frequency_counter += 0.5
                    self.absence_counter *= 0.95
                    # print(f'Frequency counter: {self.frequency_counter}')
                    # print(f'Particle counter: {count}')

                    reward = math.sqrt(self.frequency_counter)
                    # print(count, self.frequency_counter, reward, self.global_signal)
                    self.global_signal += reward / 800
                elif spike == 0:
                    input_neuron = self.get_layer(0).neurons[i+4]
                    input_neuron.input_spike(0.5)
                    self.absence_counter += 0.5

                    punishment = math.sqrt(self.absence_counter)
                    self.global_signal -= punishment / 800
                    # print(self.global_signal)
            elif i == 4:
                if spike == 1:
                    input_neuron = self.get_layer(0).neurons[i+4]
                    input_neuron.input_spike(1)
                    self.global_signal += 10
                    print(f'TAP TAP TAP')
            elif i == 5:
                if spike == 1:
                    input_neuron = self.get_layer(0).neurons[i+4]
                    input_neuron.input_spike(1)
                    self.global_signal += 10
                    print(f'TAP TAP TAP')
    
        
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.update(self.absence_counter)

        normalized_eligibility = self.normalise_eligibility()
        if normalized_eligibility == False:
            # print("No normalization performed. Skipping weight updates.")
            pass
        else:
            for layer in self.layers:
                for neuron in layer.neurons:
                    neuron.update_weights(normalized_eligibility[neuron.id], self.global_signal)
        
        output = self.get_outputs()
        return output

    def get_outputs(self):
        # Retrieve the outputs from the output layer
        output = [0, 0, 0, 0]
        spiked = False
        last_layer_index = len(self.layers) - 1
        for neuron in self.get_layer(last_layer_index).neurons:
            if neuron.spiked:
                neuron.spiked = False
                output[int(neuron.id[-1])] = 1
                spiked = True
            else:
                output[int(neuron.id[-1])] = round(neuron.membrane_potential, 2)
        # if spiked == True:
        #     if self.print_counter > 100:
        #         # print(f'Action potentials: {output}')
        #         self.print_counter = 0
        #     self.print_counter += 1

        # ACTIONS BY MEMBRANE POTENTIAL INSTEAD OF SPIKES
        
        # if spiked == False:
        #     for neuron in self.get_layer(last_layer_index).neurons:
        #         output[int(neuron.id[-1])] = neuron.membrane_potential
        #     if self.print_counter > 50:
        #         # print(f'Membrane potentials: {output}')
        #         self.print_counter = 0
        #     self.print_counter += 1
        #     max_potential = max(output)
        #     output = [1 if potential == max_potential else 0 for potential in output]
        
        return output
             
    def modify_learning(self, reward_signal):
        self.global_signal += reward_signal

    def start_trial(self):
        start_counter = 0
            
        # print(self.global_signal)
        while start_counter < 10:
            if self.global_signal != 0:
                normalized_eligibility = self.normalise_eligibility()
                if normalized_eligibility == False:
                    # print("No normalization performed. Skipping weight updates.")
                    pass
                else:
                    for layer in self.layers:
                        for neuron in layer.neurons:
                            neuron.update(self.absence_counter)
                            neuron.update_weights(normalized_eligibility[neuron.id], self.global_signal)
            start_counter += 1
            # print(f'Countdown to start: {10 - start_counter}')

        self.reset_membranes()
        self.global_signal = 0
        self.frequency_counter = 1
        self.absence_counter = 0

        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.base_activity = 0
                for key in neuron.eligibility:
                    neuron.eligibility[key] = 0

    def normalise_eligibility(self):
        total_eligibility = sum(sum(neuron.eligibility.values()) for layer in self.layers for neuron in layer.neurons)
        normalized_eligibility = {}
        if total_eligibility == 0:
            # print("Total eligibility is zero. Normalization not performed.")
            normalized_eligibility = False
            return normalized_eligibility
        for layer in self.layers:
            for neuron in layer.neurons:
                normalized_eligibility[neuron.id] = {source: trace / total_eligibility for source, trace in neuron.eligibility.items()}
        return normalized_eligibility

    def update_signal(self):
        self.global_signal *= self.signal_decay
    
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
                    if source_id == neuron.id:  # Check if the source neuron is the same as the current neuron
                        weights[f'{neuron.id}_rec'] = weight  # Store the weight with the _rec suffix
                    else:
                        weights[f'l{i-1}-n{k}_l{i}-n{j}'] = weight
        return weights
        
                


