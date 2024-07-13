import unittest
import numpy as np
import io
import logging
from evolution import Evolution
from genome import Genome, InnovationTracker  
from conversion import GenomeConverter
from network import run_xor_simulation
import networkx as nx

def setup_logger():
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return log_stream

def print_network_topology(genome):
    print("\nNetwork Topology:")
    print("Nodes:")
    for node in genome.nodes:
        print(f"  Node ID: {node['node_id']}, Type: {node['node_type']}")
    
    print("Connections:")  
    for conn in genome.connections:
        if conn.enabled:
            print(f"  From {conn.in_node} to {conn.out_node}, Weight: {conn.weight:.3f}")


class TestNEATSystem(unittest.TestCase):

    def setUp(self):
        print("\n--- Setting up test environment ---")
        self.population_size = 150
        self.n_inputs = 2
        self.n_outputs = 1  # Updated to 1 output
        self.parameters = {
            "speciation": {
                "c_one": 1.0,
                "c_two": 1.0,
                "c_three": 0.4,
                "compatibility_threshold": 1.2
            },
            "mutation": {
                "weight_mutation_rate": 0.8,
                "weight_perturbation_rate": 0.9, 
                "connection_mutation_rate": 0.1,
                "node_mutation_rate": 0.06
            },
            "reproduction": {
                "reproduction_rate": 0.75,
                "elimination_rate": 0.1,
                "interspecies_mating_rate": 0.0,  
            }
        }
        print("Initializing Evolution object...")
        self.evolution = Evolution(self.population_size, self.n_inputs, self.n_outputs, self.parameters)
        print("Setup complete.")

    def xor_fitness(self, genome):
        genome_id = genome.genome_id
        converter = GenomeConverter(genome)
        converter.convert()
        
        total_correct = 0
        num_tests = 5
        expected_outputs = [0, 1, 1, 0]
        
        for _ in range(num_tests):
            results = run_xor_simulation(converter)
            for result, expected in zip(results, expected_outputs):
                if result == expected:  # Updated to match the single output
                    total_correct += 1
        
        return total_correct / (len(expected_outputs) * num_tests)

    def test_xor_evolution(self):
        print("\n--- Testing XOR evolution ---")  
        log_stream = setup_logger()

        max_generations = 1000
        target_fitness = 1.00

        for gen in range(max_generations):
            print(f"\n\nGeneration {gen+1}/{max_generations}")
            best_genome, best_fitness = self.evolution.evolve_one_generation(self.xor_fitness)

            print(f"BEST FITNESS: {best_fitness}")
            print_network_topology(best_genome)

            if best_fitness >= target_fitness:
                print(f"Target fitness reached in generation {gen+1}")
                break

        print(f"Evolution completed. Best fitness: {best_fitness}")
        
        # Verify the best genome 
        converter = GenomeConverter(best_genome)
        converter.convert()

        print_network_topology(best_genome)
        
        print("\nFinal XOR test results:")
        for test_run in range(10):
            print(f"\nTest run {test_run + 1}:")
            results = run_xor_simulation(converter)  
            expected_outputs = [0, 1, 1, 0]
            
            for i, (result, expected) in enumerate(zip(results, expected_outputs)):
                input_pattern = ["00", "01", "10", "11"][i]
                print(f"Input: {input_pattern}, Output: {result}, Expected: {expected}")  # Updated to match the single output

        self.assertGreaterEqual(best_fitness, target_fitness, "Evolution did not reach target fitness")

    def test_network_structure(self):
        print("\n--- Testing network structure ---")
        best_genome = max(self.evolution.population.genomes, key=self.xor_fitness)
        
        converter = GenomeConverter(best_genome)  
        converter.convert()

        print(f"Number of nodes: {len(converter.node_array)}")
        print(f"Number of connections: {converter.weight_matrix.nnz}")

        # Check for cycles
        G = nx.DiGraph()
        for conn in best_genome.connections:
            if conn.enabled:
                G.add_edge(conn.in_node, conn.out_node)

        is_dag = nx.is_directed_acyclic_graph(G)
        print(f"Is the network a DAG (no cycles)? {is_dag}")

        self.assertTrue(is_dag, "The evolved network contains cycles")

        print_network_topology(best_genome)

if __name__ == '__main__':
    print("Starting NEAT System Tests")  
    unittest.main(verbosity=2)