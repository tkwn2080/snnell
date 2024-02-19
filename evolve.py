import numpy as np
import copy

def spawn():
    # self.prong_length = length
    # length = np.random.uniform(40, 100)
    length = 60

    # self.prong_angle = probe_angle
    # probe_angle = np.random.uniform(15, 75) * (np.pi / 180)
    probe_angle = 55 * (np.pi / 180)

    # self.response_angle = response_angle
    # response_angle = np.random.uniform(2, 8) * (np.pi / 180)
    response_angle = 3 * (np.pi / 180)

    # self.movement_counter = distance
    # distance = np.random.uniform(2, 8)
    distance = 3

    # self.speed = speed
    # speed = np.random.uniform(2, 8)
    speed = 3

    architecture, depth = random_architecture()

    # architecture = [10, 20, 30, 20, 16, 8, 4]
    # depth = 6

    weights = initialise_weights(architecture)

    learning_rate = np.random.uniform(0.1, 2) / 10000
    # learning_rate = 0.0001

    eligibility_decay = np.random.uniform(0.5,0.99)
    # eligibility_decay = 0.95

    ltp_rate = np.random.uniform(0.1, 5) / 1000
    ltd_rate = np.random.uniform(0.1, 5) / 10000

    parameters = []

    parameters.append(learning_rate)
    parameters.append(eligibility_decay)
    parameters.append(ltp_rate)
    parameters.append(ltd_rate)

    weights, recurrence = add_recurrence(weights, architecture)

    genetic_code = [length, probe_angle, response_angle, distance, speed, weights, architecture, depth, parameters, recurrence]

    return genetic_code

def random_architecture():
    input_layer = 10
    output_layer = 4
    architecture = []
    architecture.append(input_layer)
    hidden_depth = np.random.randint(2,4)
    for i in range(hidden_depth):
        hidden_breadth = np.random.randint(8,16)
        architecture.append(hidden_breadth)
    architecture.append(output_layer)
    print(architecture)
    total_depth = hidden_depth + 1
    return architecture, total_depth
    
def initialise_weights(dimensions):
    weights = {}
    for i in range(1, len(dimensions)):
        for j in range(dimensions[i]):
            for k in range(dimensions[i - 1]):
                weights[f'l{i-1}-n{k}_l{i}-n{j}'] = np.random.normal(0, 1)
    return weights

def add_recurrence(weights, dimensions):
    recurrent_layers = []
    probability = 0.8
    for i, _ in enumerate(dimensions[1:-1]):
        if np.random.uniform(0, 1) < probability:
            layer = i + 1
            recurrent_layers.append(layer)

            for n in range(dimensions[layer]):
                weight = np.random.normal(0, 1)
                weights[f'l{layer}-n{n}_rec'] = weight
                # print(f'Added recurrence weight {weight} to neuron {n} in layer {layer}')

            probability /= 2  # Halve the probability of adding another layer
        else:
            break
                
    return weights, recurrent_layers

def init_population(size):
    pop = []
    for i in range(size):
        pop.append(spawn())
    return pop

def mutate(individual, mutation_rate):
    mutation_strength = 0.1

    new_individual = copy.deepcopy(individual) 
    for i in range(len(new_individual)-1):
        if np.random.uniform(0, 1) < mutation_rate:
            mutation = np.random.uniform(-mutation_strength, mutation_strength)
            new_individual[i] += mutation
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
    child1 = [np.random.choice([mother[i], father[i]]) for i in range(5)]
    child2 = [np.random.choice([mother[i], father[i]]) for i in range(5)]
    child3 = [np.random.choice([mother[i], father[i]]) for i in range(5)]
    child4 = [np.random.choice([mother[i], father[i]]) for i in range(5)]

    # print(child1, child2, child3, child4)

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
            # mutated_weights = weights_mutation(mother[5])
            # mutated_child.append(mutated_weights)
            mutated_child.append(mother[5])
            # print(mother[6])
            mutated_child.append(mother[6])
            # print(mother[7])
            mutated_child.append(mother[7])
            # print(mother[8])
            mutated_child.append(mother[8])
            # print(mother[9])
            mutated_child.append(mother[9])
        else:
            # mutated_weights = weights_mutation(father[5])
            # mutated_child.append(mutated_weights)
            mutated_child.append(father[5])
            # print(father[6])
            mutated_child.append(father[6])
            # print(father[7])
            mutated_child.append(father[7])
            # print(father[8])
            mutated_child.append(father[8])
            # print(father[9])
            mutated_child.append(father[9])
        
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
    mutation_rate = 0.01
    mutation_strength = 0.001
    new_weights = copy.deepcopy(weights)
    for key in new_weights:
        if np.random.uniform(0, 1) < mutation_rate:
            mutation = np.random.uniform(-mutation_strength, mutation_strength)
            new_weights[key] += mutation
            new_weights[key] = min(max(new_weights[key], 0), 1)
    return new_weights

def neural_reproduction(individual, offspring):
    # Initialise the population, add elite
    progeny = []
    progeny.append(individual)

    for i in range(offspring - 1):
        print(f'Generating offspring {i + 1}')

        # Create a new individual
        new_individual = copy.deepcopy(individual)

        # Mutate the new individual
        new_individual[5] = weights_mutation(new_individual[5])
        progeny.append(new_individual)

    return progeny