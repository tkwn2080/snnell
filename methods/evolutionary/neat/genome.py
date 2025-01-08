import numpy as np
from ulid import ULID
import logging

logger = logging.getLogger(__name__)

initial_ids = [str(ULID()) for _ in range(100)]

class Connection:
    def __init__(self, connection_id, in_node, out_node, weight, innovation_id, enabled=True):
        self.connection_id = connection_id
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.innovation_id = innovation_id
        self._enabled = enabled

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        if self._enabled != value:
            self._enabled = value
            # print(f"Connection {self.connection_id} {'disabled' if not value else 'enabled'}. Source: Genome class")

class Genome:
    def __init__(self, n_in, n_out, innovation_tracker, source=None):
        self.genome_id = str(ULID())

        if not source:
            self.n_in = n_in
            self.n_out = n_out
            self.innovation_tracker = innovation_tracker
            self.fitness = 0

            self.nodes = []
            self.connections = []

            self.species_id = None
            self.species_counter = 0
            self.species_max = None

            # Initialize input and output nodes
            self.input_nodes = [self.add_node(initial_ids[i], 'input') for i in range(n_in)]
            self.output_nodes = [self.add_node(initial_ids[n_in + i], 'output') for i in range(n_out)]

            # Initialize connections between all input and output nodes
            innovation_number = 1
            for in_node in self.input_nodes:
                for out_node in self.output_nodes:
                    weight = np.random.uniform(-1, 1)
                    connection_id = str(ULID())
                    new_connection = Connection(connection_id, in_node, out_node, weight, innovation_number)
                    self.connections.append(new_connection)
                    innovation_number += 1

        elif source:
            self.n_in = n_in
            self.n_out = n_out
            self.innovation_tracker = innovation_tracker
            self.fitness = 0

            self.connections = [Connection(conn.connection_id, conn.in_node, conn.out_node,
                                           conn.weight, conn.innovation_id, conn.enabled)
                                for conn in source[0]]
            self.nodes = source[1]

            self.species_id = source[2]
            self.species_counter = source[2]
            self.species_max = source[3]

    def add_node(self, node_id, node_type, model="spiking_izhikevich"):
        logger.debug(f"Adding node {node_id} of type {node_type} to genome {self.genome_id}")
        if any(node['node_id'] == node_id for node in self.nodes):
            print(f"Debug: Attempting to add duplicate node {node_id} to genome")
            return None
        new_node = {'node_id': node_id, 'node_type': node_type, 'model': model}
        self.nodes.append(new_node)
        return node_id

    def add_connection(self, in_node, out_node, weight):
        logger.debug(f"Adding connection from {in_node} to {out_node} in genome {self.genome_id}")
        # Check if connection already exists
        if any(conn.in_node == in_node and conn.out_node == out_node for conn in self.connections):
            print(f"Debug: Connection from {in_node} to {out_node} already exists")
            return False

        innovation_id = self.innovation_tracker.get_innovation_number(in_node, out_node)
        connection_id = str(ULID())
        new_connection = Connection(connection_id, in_node, out_node, weight, innovation_id)
        self.connections.append(new_connection)
        return True

    def get_connection(self, connection_id):
        return next((conn for conn in self.connections if conn.connection_id == connection_id), None)

    def get_connection_by_innovation_id(self, innovation_number):
        return next((conn for conn in self.connections if conn.innovation_id == innovation_number), None)

    def get_node(self, node_id):
        return next((node for node in self.nodes if node['node_id'] == node_id), None)

    def mutate_weight(self, connection_id, new_weight):
        connection = self.get_connection(connection_id)
        if connection:
            connection.weight = new_weight

    def disable_connection(self, connection_id):
        connection = self.get_connection(connection_id)
        if connection:
            connection.enabled = False

    def enable_connection(self, connection_id):
        connection = self.get_connection(connection_id)
        if connection:
            connection.enabled = True

class InnovationTracker:
    def __init__(self):
        self.innovation_number = 0
        self.innovations = {}
        self.current_generation_innovations = {}

    def get_innovation_number(self, in_node, out_node):
        key = (in_node, out_node)
        if key in self.current_generation_innovations:
            return self.current_generation_innovations[key]
        elif key in self.innovations:
            return self.innovations[key]
        else:
            self.innovation_number += 1
            self.current_generation_innovations[key] = self.innovation_number
            return self.innovation_number

    def end_generation(self):
        self.innovations.update(self.current_generation_innovations)
        self.current_generation_innovations.clear()