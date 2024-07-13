import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from typing import Dict, Any, List, Tuple
import networkx as nx
import logging

logger = logging.getLogger(__name__)

class GenomeConverter:
    def __init__(self, genome):
        self.genome = genome
        self.node_lookup: Dict[str, Tuple[int, str]] = {}
        self.input_nodes: List[Dict] = []
        self.hidden_nodes: List[Dict] = []
        self.output_nodes: List[Dict] = []
        self.node_array: List[Dict] = []
        self.weight_matrix = None
        self.null_mask = None
        self.graph = nx.DiGraph()

        # Convert the genome to a network
        self.convert()

    def convert(self):
        self._preprocess_nodes()
        self._create_node_array()
        self._create_weight_matrix()
        self._create_null_mask()

    def get_description(self):
        description = [len(self.input_nodes), len(self.hidden_nodes), len(self.output_nodes)]
        return description

    def _preprocess_nodes(self):
        node_types = defaultdict(list)
        for node in self.genome.nodes:
            node_types[node['node_type']].append(node)

        current_index = 0
        for node_type in ['input', 'hidden', 'output']:
            for node in node_types[node_type]:
                getattr(self, f"{node_type}_nodes").append(node)
                self.node_lookup[node['node_id']] = (current_index, node_type)
                current_index += 1

        self.node_array = self.input_nodes + self.hidden_nodes + self.output_nodes
        self._validate_preprocessing()

    def _validate_preprocessing(self):
        assert len(self.node_array) == len(self.genome.nodes), "Node count mismatch"
        assert len(self.input_nodes) == self.genome.n_in, "Input node count mismatch"
        assert len(self.output_nodes) == self.genome.n_out, "Output node count mismatch"
        indices = [info[0] for info in self.node_lookup.values()]
        assert len(indices) == len(set(indices)), "Duplicate node indices found"
        assert set(indices) == set(range(len(self.node_array))), "Non-contiguous node indexing"

    def _create_node_array(self):
        node_dtype = np.dtype([
            ('index', np.int32),
            ('node_id', 'U50'),
            ('node_type', 'U10'),
            ('a', np.float32),
            ('b', np.float32),
            ('c', np.float32),
            ('d', np.float32),
            ('bias', np.float32),
            ('activation', 'U20'),
        ])

        self.node_array = np.zeros(len(self.genome.nodes), dtype=node_dtype)
        for i, node in enumerate(self.input_nodes + self.hidden_nodes + self.output_nodes):
            self.node_array[i] = self._create_node_entry(i, node)

        self._validate_node_array()

    def _create_node_entry(self, index: int, node: Dict[str, Any]) -> tuple:
        return (
            index,
            node['node_id'],
            node['node_type'],
            node.get('a', 0.02),  # 0.02 is the default value
            node.get('b', 0.2),  # 0.2 is the default value
            node.get('c', -65),  # -65 is the default value
            node.get('d', 8),  # 8 is the default value
            node.get('bias', 0.0),
            node.get('activation', 'None')
        )

    def _validate_node_array(self):
        assert len(self.node_array) == len(self.genome.nodes), "Node count mismatch in final array"
        type_counts = {
            'input': np.sum(self.node_array['node_type'] == 'input'),
            'hidden': np.sum(self.node_array['node_type'] == 'hidden'),
            'output': np.sum(self.node_array['node_type'] == 'output')
        }
        assert type_counts['input'] == len(self.input_nodes), "Input node count mismatch"
        assert type_counts['hidden'] == len(self.hidden_nodes), "Hidden node count mismatch"
        assert type_counts['output'] == len(self.output_nodes), "Output node count mismatch"
        assert np.all(self.node_array['index'] == np.arange(len(self.node_array))), "Index mismatch"
        assert not np.any(np.isnan(self.node_array['a'])), "Invalid 'a' parameter found"
        assert not np.any(np.isnan(self.node_array['b'])), "Invalid 'b' parameter found"
        assert not np.any(np.isnan(self.node_array['c'])), "Invalid 'c' parameter found"
        assert not np.any(np.isnan(self.node_array['d'])), "Invalid 'd' parameter found"

    def _create_weight_matrix(self):
        num_nodes = len(self.node_array)
        row_indices, col_indices, weights = [], [], []

        for connection in self.genome.connections:
            if connection.enabled:
                from_index = self.node_lookup[connection.in_node][0]
                to_index = self.node_lookup[connection.out_node][0]
                row_indices.append(to_index)
                col_indices.append(from_index)
                weights.append(connection.weight)
                self.graph.add_edge(connection.in_node, connection.out_node)

        self.weight_matrix = csr_matrix((weights, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
        self._validate_weight_matrix()

    def _validate_weight_matrix(self):
        assert self.weight_matrix.shape == (len(self.node_array), len(self.node_array)), "Weight matrix dimensions mismatch"
        input_connections = self.weight_matrix[:len(self.input_nodes), :].nnz
        assert input_connections == 0, f"Found {input_connections} invalid connections to input nodes"
        assert np.all(self.weight_matrix.diagonal() == 0), "Self-connections detected"
        enabled_connections = sum(1 for conn in self.genome.connections if conn.enabled)
        assert self.weight_matrix.nnz == enabled_connections, f"Mismatch in connection count. Matrix: {self.weight_matrix.nnz}, Genome: {enabled_connections}"
        self._detect_cycles()

    def _detect_cycles(self):
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                logger.error(f"Detected {len(cycles)} cycle(s) in the network")
                raise ValueError("Network contains cycles and is not feed-forward.")
        except nx.NetworkXNoCycle:
            pass

    def _create_null_mask(self):
        num_nodes = len(self.node_array)
        num_inputs = len(self.input_nodes)
        self.null_mask = np.zeros((num_nodes, num_nodes), dtype=bool)
        self.null_mask[num_inputs:, :] = True
        np.fill_diagonal(self.null_mask, False)
        self.null_mask = self.null_mask & (self.weight_matrix.toarray() != 0)
        self.null_mask = csr_matrix(self.null_mask)
        self._validate_null_mask()

    def _validate_null_mask(self):
        assert self.null_mask.shape == (len(self.node_array), len(self.node_array)), "Null mask dimensions mismatch"
        input_connections = self.null_mask[:len(self.input_nodes), :].nnz
        assert input_connections == 0, f"Found {input_connections} invalid connections to input nodes"
        assert np.all(self.null_mask.diagonal() == 0), "Self-connections detected in null mask"
        weight_connections = set(zip(*self.weight_matrix.nonzero()))
        mask_connections = set(zip(*self.null_mask.nonzero()))
        if not weight_connections.issubset(mask_connections):
            logger.error("Some connections in weight matrix are not allowed by null mask")
            raise AssertionError("Some connections in weight matrix are not allowed by null mask")

    def print_network_structure(self):
        print("\nNetwork Structure:")
        print(f"Total nodes: {len(self.node_array)}")
        print(f"Input nodes: {len(self.input_nodes)}")
        print(f"Hidden nodes: {len(self.hidden_nodes)}")
        print(f"Output nodes: {len(self.output_nodes)}")
        print(f"Total connections: {self.weight_matrix.nnz}")
        
        # print("\nConnections:")
        for i, j in zip(*self.weight_matrix.nonzero()):
            from_node = self.node_array[j]['node_id']
            to_node = self.node_array[i]['node_id']
            weight = self.weight_matrix[i, j]
            # print(f"{from_node} -> {to_node} (weight: {weight})")

        logger.info(f"Total nodes: {len(self.node_array)}")
        logger.info(f"Input nodes: {len(self.input_nodes)}")
        logger.info(f"Hidden nodes: {len(self.hidden_nodes)}")
        logger.info(f"Output nodes: {len(self.output_nodes)}")
        logger.info(f"Total connections: {self.weight_matrix.nnz}")

        logger.info("Connections:")
        for i, j in zip(*self.weight_matrix.nonzero()):
            from_node = self.node_array[j]['node_id']
            to_node = self.node_array[i]['node_id']
            weight = self.weight_matrix[i, j]
            logger.info(f"{from_node} -> {to_node} (weight: {weight})")

        logger.info("Node Layers:")
        layers = nx.topological_generations(self.graph)
        for i, layer in enumerate(layers):
            logger.info(f"Layer {i}: {', '.join(layer)}")

    def visualize_network(self, filename='network_structure.png'):
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(12, 8))
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
        edge_labels = {(u, v): f"{self.weight_matrix[self.node_lookup[v][0], self.node_lookup[u][0]]:.2f}" for u, v in self.graph.edges()}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
        plt.title("Network Structure")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        logger.info(f"Network visualization saved to {filename}")