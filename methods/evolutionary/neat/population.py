import numpy as np
from collections import defaultdict
from ulid import ULID
import random
import networkx as nx
import logging

from methods.evolutionary.neat.genome import Genome, InnovationTracker, Connection

logger = logging.getLogger(__name__)

def detect_cycle(connections):
    G = nx.DiGraph()
    for conn in connections:
        if conn.enabled:
            G.add_edge(conn.in_node, conn.out_node)
    try:
        cycle = nx.find_cycle(G, orientation="original")
        return [edge[0] for edge in cycle]  # Return only the nodes in the cycle
    except nx.NetworkXNoCycle:
        return None

class Population:
    def __init__(self, size, n_in, n_out, evolution_parameters):
        self.size = size
        self.n_in = n_in
        self.n_out = n_out

        # Initialize parameters
        self.mutation_parameters = evolution_parameters["mutation"]
        self.speciation_parameters = evolution_parameters["speciation"]
        self.reproduction_parameters = evolution_parameters["reproduction"]

        # Initialize auxiliaries
        self.innovation_tracker = InnovationTracker()

        # Initialize population
        self.genomes = [Genome(n_in, n_out, self.innovation_tracker) for _ in range(size)]
        self.generation = 0
        print(f"Initialized population with {size} genomes")

    def replace_population(self, population):
        self.genomes = population
        self.innovation_tracker.end_generation()

class Speciation:
    def __init__(self, speciation_parameters):
        self.species = {}
        self.c_one = speciation_parameters["c_one"] # Coefficient for excess difference
        self.c_two = speciation_parameters["c_two"] # Coefficient for disjoint difference
        self.c_three = speciation_parameters["c_three"] # Coefficient for weight difference
        self.compatibility_threshold = speciation_parameters["compatibility_threshold"] # Threshold for compatibility (exceeding this causes speciation)

    def speciate(self, population):
        new_species = {}
        next_species_id = 0
        
        # Choose a random genome to start speciation
        random_genome = random.choice(population)
        first_species_id = next_species_id
        new_species[first_species_id] = [random_genome]
        random_genome.species_id = first_species_id
        next_species_id += 1
        
        for genome in population:
            if genome == random_genome:
                continue  # Skip the already assigned random genome
            
            assigned_to_species = False
            for species_id, members in new_species.items():
                representative = members[0]  # Use the first member as representative
                compatibility = self.determine_compatibility(genome, representative)
                if compatibility <= self.compatibility_threshold:
                    new_species[species_id].append(genome)
                    genome.species_id = species_id
                    assigned_to_species = True
                    break
            
            if not assigned_to_species:
                new_species_id = next_species_id
                new_species[new_species_id] = [genome]
                genome.species_id = new_species_id
                next_species_id += 1

        self.species = {k: v for k, v in new_species.items() if v}
        return self.species

    def align_innovations(self, first_genome, second_genome):
        first_innovations = [conn.innovation_id for conn in first_genome.connections if conn.enabled]
        second_innovations = [conn.innovation_id for conn in second_genome.connections if conn.enabled]

        excess_diff = self.excess_difference(first_innovations, second_innovations)
        disjoint_diff = self.disjoint_difference(first_innovations, second_innovations)
        weight_diff = self.weight_difference(first_genome, second_genome, first_innovations, second_innovations)

        return excess_diff, disjoint_diff, weight_diff

    def excess_difference(self, first_innovations, second_innovations):
        if not first_innovations and not second_innovations:
            return 0
        max_innovation = max(max(first_innovations or [0]), max(second_innovations or [0]))
        return sum(1 for i in first_innovations + second_innovations if i > max_innovation)

    def disjoint_difference(self, first_innovations, second_innovations):
        if not first_innovations or not second_innovations:
            return len(first_innovations) + len(second_innovations)
        min_innovation = min(max(first_innovations), max(second_innovations))
        return len([i for i in first_innovations + second_innovations if i <= min_innovation]) - len(set(first_innovations) & set(second_innovations))

    def weight_difference(self, first_genome, second_genome, first_innovations, second_innovations):
        matching_innovations = set(first_innovations) & set(second_innovations)
        if not matching_innovations:
            return 0

        weight_diff = 0
        for i in matching_innovations:
            first_conn = first_genome.get_connection_by_innovation_id(i)
            second_conn = second_genome.get_connection_by_innovation_id(i)

            if first_conn is None:
                logger.warning(f"Connection with innovation ID {i} not found in the first genome.")
            if second_conn is None:
                logger.warning(f"Connection with innovation ID {i} not found in the second genome.")

            if first_conn is not None and second_conn is not None:
                weight_diff += abs(first_conn.weight - second_conn.weight)
            else:
                weight_diff += 1

        return weight_diff / len(matching_innovations)

    def determine_compatibility(self, first_genome, second_genome):
        E, D, W = self.align_innovations(first_genome, second_genome)
        
        # Normalize by dividing by the number of genes in the larger genome
        N = max(len(first_genome.connections), len(second_genome.connections))
        N = 1 if N == 0 else N  # Avoid division by zero
        
        return (self.c_one * E / N) + (self.c_two * D / N) + (self.c_three * W)

class Reproduction:
    def __init__(self, reproduction_parameters):
        self.elimination_rate = reproduction_parameters["elimination_rate"]
        self.reproduction_rate = reproduction_parameters["reproduction_rate"]
        self.interspecies_mating_rate = reproduction_parameters["interspecies_mating_rate"]
        self.elitism_count = 2  # Number of top individuals to preserve in each species

        self.stagnation_limit = 12

    def reproduce(self, population, species, fitness):
        new_population = []

        # Handle species stagnation
        for species_id, members in species.items():
            # Find the best fitness in the current species
            current_best_fitness = min(fitness[genome.genome_id] for genome in members)
            
            # If species_max is unassigned, initialize it
            if members[0].species_max is None:
                for genome in members:
                    genome.species_max = current_best_fitness
                    genome.species_counter = 0
            else:
                # Check if the current set has a better maximum fitness
                if current_best_fitness < members[0].species_max:
                    # New best fitness found
                    for genome in members:
                        genome.species_max = current_best_fitness
                        genome.species_counter = 0
                else:
                    # No improvement, increment stagnation counter
                    for genome in members:
                        genome.species_counter += 1

        # Step 1: Apply elitism and eliminate low-performing individuals
        species = self.elitism_and_elimination(species, fitness)
        
        # Step 2: Compute the number of offspring for each species
        total_adjusted_fitness = sum(self.compute_adjusted_fitness(s, fitness) for s in species.values())
        species_offspring = self.compute_offspring(species, fitness, total_adjusted_fitness, len(population))
        
        # Step 3: Produce offspring for each species
        for species_id, offspring_count in species_offspring.items():
            species_members = species[species_id]
            
            # Check for stagnation
            if species_members[0].species_counter >= self.stagnation_limit:
                # Only allow reproduction if the species is performing well
                species_fitness = min(fitness[genome.genome_id] for genome in species_members)
                if species_fitness >= 500:  # This threshold might need adjustment
                    print(f"Species {species_id} is stagnant but performing well. Allowing reproduction.")
                else:
                    print(f"Species {species_id} is stagnant and underperforming. Preventing reproduction.")
                    continue  # Skip to the next species

            # Sort species members by fitness (lower is better)
            sorted_members = sorted(species_members, key=lambda genome: fitness[genome.genome_id])
            
            # Add elite members to new population
            new_population.extend(sorted_members[:min(self.elitism_count, len(sorted_members))])
            
            # Produce remaining offspring
            for _ in range(offspring_count - min(self.elitism_count, len(sorted_members))):
                if np.random.rand() < self.interspecies_mating_rate:
                    child = self.interspecies_reproduction(population, species_members)
                else:
                    child = self.intraspecies_reproduction(species_members)
                new_population.append(child)

        # Step 4: Make up for lost individuals by producing more offspring from the fittest species
        while len(new_population) < len(population):
            fittest_species = max(species.values(), key=lambda s: self.compute_adjusted_fitness(s, fitness))
            new_population.append(self.intraspecies_reproduction(fittest_species))
        
        return new_population

    def elitism_and_elimination(self, species, fitness):
        for species_id, members in species.items():
            # Sort members by fitness (lower is better)
            sorted_members = sorted(members, key=lambda genome: fitness[genome.genome_id])
            
            if len(sorted_members) == 1:
                species[species_id] = sorted_members
            else:
                # Keep the top performers (elites) and eliminate the rest
                keep_count = max(self.elitism_count, int(len(sorted_members) * (1 - self.elimination_rate)))
                species[species_id] = sorted_members[:keep_count]
        
        return species

    def compute_adjusted_fitness(self, species_members, fitness):
        # Use inverse of fitness for adjustment (lower original fitness = higher adjusted fitness)
        return sum((1 / fitness[genome.genome_id]) / len(species_members) for genome in species_members)

    def compute_offspring(self, species, fitness, total_adjusted_fitness, population_size):
        species_offspring = {}
        for species_id, members in species.items():
            species_adjusted_fitness = self.compute_adjusted_fitness(members, fitness)
            species_offspring[species_id] = max(1, int((species_adjusted_fitness / total_adjusted_fitness) * population_size))
        return species_offspring

    def intraspecies_reproduction(self, species_members):
        if len(species_members) == 1:
            return species_members[0]
        
        # Select parents randomly
        parent1, parent2 = random.sample(species_members, 2)
        
        return self.crossover(parent1, parent2)

    def interspecies_reproduction(self, population, species_members):
        # Choose first parent from the species randomly
        parent1 = random.choice(species_members)
        
        # Choose second parent from the entire population, excluding the species of parent1
        other_species = [g for g in population if g not in species_members]
        parent2 = random.choice(other_species)
        
        return self.crossover(parent1, parent2)

    def crossover(self, parent1, parent2): # This will need to be rewritten entirely
        logger.debug(f"Starting crossover between genomes {parent1.genome_id} and {parent2.genome_id}")

        # Determine which parent is more fit (lower fitness is better)
        if parent1.fitness < parent2.fitness:
            fit_parent, less_fit_parent = parent1, parent2
        else:
            fit_parent, less_fit_parent = parent2, parent1

        print(f"More fit parent has {len(fit_parent.connections)} connections and {len(fit_parent.nodes)} nodes")
        print(f"Less fit parent has {len(less_fit_parent.connections)} connections and {len(less_fit_parent.nodes)} nodes")

        # Create a new child genome
        child_connections = []
        child_nodes = {}  # Use a dictionary to store nodes

        # Sort connections by innovation ID
        fit_parent_connections = sorted(fit_parent.connections, key=lambda conn: conn.innovation_id)
        less_fit_parent_connections = sorted(less_fit_parent.connections, key=lambda conn: conn.innovation_id)

        # Find the last matching innovation
        last_matching_innovation = 0
        matching_innovations = 0
        for fp_conn, lfp_conn in zip(fit_parent_connections, less_fit_parent_connections):
            if fp_conn.innovation_id == lfp_conn.innovation_id:
                last_matching_innovation = fp_conn.innovation_id
                matching_innovations += 1
            else:
                break

        print(f'Couple has {matching_innovations} matching innovations')

        # Helper function to add a node to child_nodes
        def add_node(node_id, parent):
            if node_id not in child_nodes:
                node = next((n for n in parent.nodes if n['node_id'] == node_id), None)
                if node:
                    child_nodes[node_id] = node.copy()  # Create a copy of the node

        # Crossover up to the last matching innovation
        for fp_conn in fit_parent_connections:
            if fp_conn.innovation_id <= last_matching_innovation:
                lfp_conn = next((conn for conn in less_fit_parent_connections if conn.innovation_id == fp_conn.innovation_id), None)
                if lfp_conn and random.random() < 0.5:
                    selected_conn = lfp_conn
                    parent = less_fit_parent
                else:
                    selected_conn = fp_conn
                    parent = fit_parent
                
                # Only add the connection if it's enabled
                if selected_conn.enabled:
                    child_connections.append(Connection(
                        selected_conn.connection_id,
                        selected_conn.in_node,
                        selected_conn.out_node,
                        selected_conn.weight,
                        selected_conn.innovation_id,
                        selected_conn.enabled
                    ))
                    add_node(selected_conn.in_node, parent)
                    add_node(selected_conn.out_node, parent)
            else:
                # Add remaining connections from the fit parent, but only if they're enabled
                if fp_conn.enabled:
                    child_connections.append(Connection(
                        fp_conn.connection_id,
                        fp_conn.in_node,
                        fp_conn.out_node,
                        fp_conn.weight,
                        fp_conn.innovation_id,
                        fp_conn.enabled
                    ))
                    add_node(fp_conn.in_node, fit_parent)
                    add_node(fp_conn.out_node, fit_parent)

        # print(f'Child now has {len(child_connections)} connections')

        # Convert child_nodes dictionary to a list
        child_nodes_list = list(child_nodes.values())

        # Create a new child genome using the assembled connections and nodes
        child = Genome(parent1.n_in, parent1.n_out, parent1.innovation_tracker, source=[child_connections, child_nodes_list, parent1.species_id, parent1.species_counter, parent1.species_max])

        # print(f"Child has {len(child.connections)} connections and {len(child.nodes)} nodes")

        # Check for cycles (if you have a detect_cycle function)
        # cycle = detect_cycle(child.connections)
        # if cycle:
        #     print(f"Cycle detected in crossover: {' -> '.join(cycle)}")

        if len(child.nodes) < len(parent1.nodes) or len(child.nodes) < len(parent2.nodes):
            print(f"Child has fewer nodes ({len(child.nodes)}) than parents ({len(parent1.nodes)} and {len(parent2.nodes)})")

        return child

class Mutation:
    def __init__(self, mutation_parameters):
        self.weight_mutation_rate = mutation_parameters["weight_mutation_rate"]
        self.weight_perturbation_rate = mutation_parameters["weight_perturbation_rate"]
        self.connection_mutation_rate = mutation_parameters["connection_mutation_rate"]
        self.node_mutation_rate = mutation_parameters["node_mutation_rate"]

    def mutate(self, genome):
        logger.debug(f"Starting mutation for genome {genome.genome_id}")
        mutated_genome = Genome(genome.n_in, genome.n_out, genome.innovation_tracker, source=[genome.connections, genome.nodes, genome.species_id, genome.species_counter, genome.species_max])
        if np.random.rand() < self.connection_mutation_rate:
            # print("Attempting connection mutation")
            logger.debug("Attempting connection mutation")
            self.connection_mutation(mutated_genome)
        if np.random.rand() < self.node_mutation_rate:
            logger.debug("Attempting node mutation")
            self.node_mutation(mutated_genome)
        self.weight_mutation(mutated_genome) # Should this be before or after the node/connection mutations?

        cycle = detect_cycle(mutated_genome.connections)
        if cycle:
            logger.warning(f"Cycle detected after mutation: {' -> '.join(cycle)}")

        logger.debug(f"Mutation complete for genome {genome.genome_id}")
        return mutated_genome

    def weight_mutation(self, genome):
        for connection in genome.connections:
            if connection.enabled and np.random.rand() < self.weight_mutation_rate:
                old_weight = connection.weight
                if np.random.rand() < self.weight_perturbation_rate:
                    connection.weight += np.random.uniform(-0.1, 0.1)
                else:
                    connection.weight = np.random.uniform(-1, 1)
                connection.weight = np.clip(connection.weight, -1, 1)
                logger.debug(f"Mutated weight from {old_weight} to {connection.weight}")

    def connection_mutation(self, genome):
        logger.debug(f"Starting connection mutation for genome {genome.genome_id}")
        available_connections = self.get_available_connections(genome)
        logger.debug(f"Available connections: {len(available_connections)}")

        if available_connections:
            in_node, out_node = random.choice(available_connections)
            logger.debug(f"Selected connection: {in_node['node_id']} -> {out_node['node_id']}")

            new_weight = np.random.uniform(-1, 1)
            enabled = True
            success = genome.add_connection(in_node['node_id'], out_node['node_id'], new_weight)
            
            if success:
                logger.debug(f"Added new connection: {in_node['node_id']} -> {out_node['node_id']}")
                print(f"Added new connection: {in_node['node_id']} -> {out_node['node_id']}")
                cycle = detect_cycle(genome.connections)
                if cycle:
                    logger.warning(f"Cycle detected in connection mutation: {' -> '.join(cycle)}")
            else:
                logger.warning(f"Failed to add new connection: {in_node['node_id']} -> {out_node['node_id']}")
        else:
            logger.warning("No available connections")

    def get_available_connections(self, genome):
        logger.debug(f"Getting available connections for genome {genome.genome_id}")
        G = nx.DiGraph()
        print(f"Genome has {len(genome.nodes)} nodes")
        enabled_connections = [conn for conn in genome.connections if conn.enabled]
        print(f"Genome has {len(enabled_connections)} enabled connections")

        for node in genome.nodes:
                G.add_node(node['node_id'])
        for conn in genome.connections:
            if conn.enabled:
                G.add_edge(conn.in_node, conn.out_node)

        if len(G.nodes()) == 0 and len(genome.nodes) > 0:
            logger.warning("Graph is empty but genome has nodes. Rebuilding graph.")
            for node in genome.nodes:
                G.add_node(node['node_id'])
            for conn in genome.connections:
                if conn.enabled:
                    G.add_edge(conn.in_node, conn.out_node)

        logger.debug(f"Graph nodes: {G.nodes()}")
        logger.debug(f"Graph edges: {G.edges()}")

        logger.debug("Genome nodes:")
        for node in genome.nodes:
            logger.debug(f"Node ID: {node['node_id']}, Type: {node['node_type']}")

        input_nodes = [node for node in genome.nodes if node['node_type'] == 'input']
        hidden_nodes = [node for node in genome.nodes if node['node_type'] == 'hidden']
        output_nodes = [node for node in genome.nodes if node['node_type'] == 'output']

        logger.debug(f"Input nodes: {[node['node_id'] for node in input_nodes]}")
        logger.debug(f"Hidden nodes: {[node['node_id'] for node in hidden_nodes]}")
        logger.debug(f"Output nodes: {[node['node_id'] for node in output_nodes]}")

        possible_connections = []

        # Input to hidden and output
        for in_node in input_nodes:
            for hid_node in hidden_nodes:
                possible_connections.append((in_node, hid_node))
            for out_node in output_nodes:
                possible_connections.append((in_node, out_node))

        # Hidden to hidden and output
        for hid_node1 in hidden_nodes:
            for hid_node2 in hidden_nodes:
                if hid_node1 != hid_node2:
                    if hid_node2['node_id'] not in G.nodes() or hid_node1['node_id'] not in G.nodes():
                        logger.warning(f"Node not in graph: {hid_node1['node_id']} or {hid_node2['node_id']}")
                        logger.debug(f"hid_node1: {hid_node1}")
                        logger.debug(f"hid_node2: {hid_node2}")
                        logger.debug(f"Is hid_node1 in genome nodes? {any(node['node_id'] == hid_node1['node_id'] for node in genome.nodes)}")
                        logger.debug(f"Is hid_node2 in genome nodes? {any(node['node_id'] == hid_node2['node_id'] for node in genome.nodes)}")
                    elif not nx.has_path(G, hid_node2['node_id'], hid_node1['node_id']):
                        possible_connections.append((hid_node1, hid_node2))
            for out_node in output_nodes:
                possible_connections.append((hid_node1, out_node))

        existing_connections = [(conn.in_node, conn.out_node) for conn in genome.connections]
        available_connections = [conn for conn in possible_connections if (conn[0]['node_id'], conn[1]['node_id']) not in existing_connections]

        logger.debug(f"Possible connections: {len(possible_connections)}")
        logger.debug(f"Existing connections: {len(existing_connections)}")
        logger.debug(f"Available connections: {len(available_connections)}")

        return available_connections

    def node_mutation(self, genome):
        logger.debug(f"Starting node mutation for genome {genome.genome_id}")
        logger.debug(f"Nodes before mutation: {[node['node_id'] for node in genome.nodes]}")
        enabled_connections = [conn for conn in genome.connections if conn.enabled]
        if not enabled_connections:
            logger.debug("No enabled connections available for node mutation")
            return

        connection = random.choice(enabled_connections)
        logger.debug(f"Selected connection for splitting: {connection.in_node} -> {connection.out_node}")

        connection_id = connection.connection_id
        genome.disable_connection(connection_id)

        new_node_id = str(ULID())
        new_node = genome.add_node(new_node_id, 'hidden')
        if new_node is None:
            logger.warning("Failed to add new node")
            return

        in_node = connection.in_node
        out_node = connection.out_node

        success1 = genome.add_connection(in_node, new_node_id, 1.0)
        success2 = genome.add_connection(new_node_id, out_node, connection.weight)

        if success1 and success2:
            logger.debug(f"Successfully added new node {new_node_id} and connections")
            cycle = detect_cycle(genome.connections)
            if cycle:
                logger.warning(f"Cycle detected in node mutation: {' -> '.join(cycle)}")
            print(f'Successfully added new node to genome {genome.genome_id}')
        else:
            logger.warning("Failed to add one or both new connections")
            genome.enable_connection(connection_id)
        logger.debug(f"Nodes after mutation: {[node['node_id'] for node in genome.nodes]}")

        enabled_connections = [conn for conn in genome.connections if conn.enabled]
        disabled_connections = [conn for conn in genome.connections if not conn.enabled]

    def neuron_mutation(self, genome):
        # If there is at least one hidden node, mutate the neuron type
        hidden_nodes = [node for node in genome.nodes if node['node_type'] == 'hidden']
        if hidden_nodes:
            # Choose a random hidden node
            hidden_node = random.choice(hidden_nodes)
            
            # Choose a random neuron type
            neuron_type = random.choice(['RS', 'IB', 'CH', 'FS', 'LTS', 'RZ'])
            
            # Change the neuron type
            hidden_node['node_type'] = neuron_type

            print(f"Mutated neuron type of hidden node {hidden_node['node_id']} to {neuron_type}")
        else:
            print("No hidden nodes found for neuron mutation")