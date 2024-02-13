# This will define the variables for each generation
# and the functions to evolve the population

import numpy as np
import copy

# We will begin with randomly initialised variables
# The set are: probe length, probe angle, angle of response (R), angle of response (L)

# How long are the probes?
length = None

# At what angle are the probes set?
probe_angle = None

# What is the angle of response?
response_angle = None

# How far will the entity move per detection?
distance = None

# How fast will the entity move?
speed = None

weights = None

genetic_code = [length, probe_angle, response_angle, distance, speed, weights]

def spawn():
    # This will create a new individual with random values
    # genetic_code = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform (0,1), np.random.uniform(0, 1), np.random.uniform(0, 1)]
    
    # self.prong_length = length
    # length = np.random.uniform(40, 100)
    length = 40

    # self.prong_angle = probe_angle
    # probe_angle = np.random.uniform(15, 75) * (np.pi / 180)
    probe_angle = 55 * (np.pi / 180)

    # self.response_angle = response_angle
    # response_angle = np.random.uniform(2, 8) * (np.pi / 180)
    response_angle = 3 * (np.pi / 180)

    # self.movement_counter = distance
    # distance = np.random.uniform(2, 8)
    distance = 5

    # self.speed = speed
    # speed = np.random.uniform(2, 8)
    speed = 5

    architecture, depth = random_architecture()

    weights = initialise_weights(architecture)

    learning_rate = np.random.uniform(0.01, 2) / 10000

    eligibility_decay = np.random.uniform(0.1,0.99)

    parameters = []

    parameters.append(learning_rate)
    parameters.append(eligibility_decay)

    genetic_code = [length, probe_angle, response_angle, distance, speed, weights, architecture, depth, parameters]

    return genetic_code

def random_architecture():
    input_layer = 4
    output_layer = 3
    architecture = []
    architecture.append(input_layer)
    hidden_depth = np.random.randint(2,8)
    for i in range(hidden_depth):
        hidden_breadth = np.random.randint(4,24)
        architecture.append(hidden_breadth)
    architecture.append(output_layer)
    print(architecture)
    total_depth = hidden_depth + 1
    return architecture, total_depth
    


def initialise_weights(dimensions):
    print(dimensions)
    # This will take the network dimensions and create a set of randomised weights
    # They will be stored as addressible weights corresponding to each connection
    # They will be slotted in to the model at the start of each candidate trial
    # The dimensions will be: 4, 5, 3
    # It will be a dense network
    # The weights will be stored in a list of lists
    # They will be initialised between 0 and 1

    weights = {}
    for i in range(1, len(dimensions)):
        fan_in = dimensions[i - 1]
        fan_out = dimensions[i]
        # Xavier/Glorot initialization variance
        variance = 2 / (fan_in + fan_out)
        layer_weights = np.random.normal(0, np.sqrt(variance), (dimensions[i], dimensions[i - 1]))
        for j in range(dimensions[i]):
            for k in range(dimensions[i - 1]):
                weights[f'l{i-1}-n{k}_l{i}-n{j}'] = layer_weights[j, k]
    return weights



def init_population(size):
    # This will create a population of size 'size'
    pop = []
    for i in range(size):
        pop.append(spawn())
    return pop



def mutate(individual, mutation_rate):
    mutation_strength = 0.1

    new_individual = copy.deepcopy(individual)  # Create a copy to mutate
    for i in range(len(new_individual)-1):
        if np.random.uniform(0, 1) < mutation_rate:
            # Apply a mutation around the parent's value within a certain range defined by mutation_strength
            mutation = np.random.uniform(-mutation_strength, mutation_strength)
            new_individual[i] += mutation
            # Ensure the new value is within the allowed range (0, 1)
            new_individual[i] = min(max(new_individual[i], 0), 1)
    
    # For now, just pass through without mutation
    # return new_individual
    return individual

def asexual_reproduction(parent, mutation_rate):
    # for n in range(progeny):
    #     offspring = []
    #     for n in range(1, progeny + 1):
    #         child = mutate(parent, mutation_rate)
    #         offspring.append((f'child{n}', child))
    #     return offspring
    
    child1 = mutate(parent, mutation_rate)
    child2 = mutate(parent, mutation_rate)
    child3 = mutate(parent, mutation_rate)
    child4 = mutate(parent, mutation_rate)
    return child1, child2, child3, child4

def sexual_reproduction(mother, father):

    # Create children by choosing attributes from either mother or father for the first five indices
    child1 = [np.random.choice([mother[i], father[i]]) for i in range(5)]
    child2 = [np.random.choice([mother[i], father[i]]) for i in range(5)]
    child3 = [np.random.choice([mother[i], father[i]]) for i in range(5)]
    child4 = [np.random.choice([mother[i], father[i]]) for i in range(5)]

    # print(child1, child2, child3, child4)

    # Append the weights reproduction separately to ensure it is placed at the 6th index
    # child1.append(weights_reproduction(mother, father))
    # child2.append(weights_reproduction(mother, father))
    # child3.append(weights_reproduction(mother, father))
    # child4.append(weights_reproduction(mother, father))

    # print(child1, child2, child3, child4)
    return child1, child2, child3, child4

def reproduction(mother, father, mutation_rate):
    progeny = sexual_reproduction(mother, father)
    mutated_progeny = []
    for child in progeny:
        mutated_child = mutate(child, mutation_rate)
        if np.random.uniform(0, 1) < 0.5:
            mutated_weights = weights_mutation(mother[5])
            mutated_child.append(mutated_weights)
            mutated_child.append(mother[6])
            mutated_child.append(mother[7])
            mutated_child.append(mother[8])
        else:
            mutated_weights = weights_mutation(father[5])
            mutated_child.append(mutated_weights)
            mutated_child.append(father[6])
            mutated_child.append(father[7])
            mutated_child.append(father[8])
        
        mutated_progeny.append(mutated_child)
    return mutated_progeny

# def weights_reproduction(mother, father):
#     mother_weights = mother[5]
#     father_weights = father[5]
#     child_weights = {}
#     for key in mother_weights:
#         child_weights[key] = np.random.choice([mother_weights[key], father_weights[key]])
#     return child_weights

def weights_mutation(weights):
    mutation_rate = 0.1
    mutation_strength = 0.001
    new_weights = copy.deepcopy(weights)
    for key in new_weights:
        if np.random.uniform(0, 1) < mutation_rate:
            mutation = np.random.uniform(-mutation_strength, mutation_strength)
            new_weights[key] += mutation
            new_weights[key] = min(max(new_weights[key], 0), 1)
    return new_weights