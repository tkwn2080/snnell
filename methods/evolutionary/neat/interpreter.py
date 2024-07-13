import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from methods.evolutionary.neat.network import NEATSNN

class NetworkConverter:
    def __init__(self, genome):
        self.genome = genome
        self.graph = nx.DiGraph()
        self.layers = defaultdict(list)
        self.node_layer_map = {}

    def build_graph(self):
        for node in self.genome.nodes:
            self.graph.add_node(node['node_id'], type=node['node_type'])
        for conn in self.genome.connections:
            if conn['enabled']:
                self.graph.add_edge(conn['in_node'], conn['out_node'], weight=conn['weight'])

    def topological_sort(self):
        try:
            sorted_nodes = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("The genome contains cycles and cannot be converted to a feedforward network.")

        # Assign layers
        for node in sorted_nodes:
            node_type = self.graph.nodes[node]['type']
            if node_type == 'input':
                self.layers[0].append(node)
                self.node_layer_map[node] = 0
            elif node_type == 'output':
                output_layer = max(self.layers.keys(), default=0) + 1
                self.layers[output_layer].append(node)
                self.node_layer_map[node] = output_layer
            else:
                # For hidden nodes, place them in the earliest possible layer
                earliest_layer = max([self.node_layer_map[pred] for pred in self.graph.predecessors(node)], default=-1) + 1
                self.layers[earliest_layer].append(node)
                self.node_layer_map[node] = earliest_layer

    def get_layer_structure(self):
        return {layer: [self.graph.nodes[node] for node in nodes] for layer, nodes in self.layers.items()}

    def get_connections(self):
        return [
            {
                'from': conn['in_node'],
                'to': conn['out_node'],
                'weight': conn['weight'],
                'from_layer': self.node_layer_map[conn['in_node']],
                'to_layer': self.node_layer_map[conn['out_node']]
            }
            for conn in self.genome.connections if conn['enabled']
        ]

    def visualize_network(self, filename='network_visualization.png'):
        if not self.layers:
            self.convert()

        plt.figure(figsize=(12, 8))
        
        # Compute positions
        pos = {}
        layer_widths = [len(layer) for layer in self.layers.values()]
        max_width = max(layer_widths)
        
        for layer, nodes in self.layers.items():
            layer_width = len(nodes)
            for i, node in enumerate(nodes):
                x = layer
                y = (i - (layer_width - 1) / 2) / max_width
                pos[node] = (x, y)

        # Draw nodes
        colors = ['lightblue' if self.graph.nodes[node]['type'] == 'input' else
                  'lightgreen' if self.graph.nodes[node]['type'] == 'output' else
                  'lightgray' for node in self.graph.nodes()]
        nx.draw_networkx_nodes(self.graph, pos, node_color=colors, node_size=500)

        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True, arrowsize=20)

        # Add labels
        labels = {node: f"{node}\n({self.graph.nodes[node]['type']})" for node in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)

        plt.title("Evolved Network")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Network visualization saved as {filename}")

        # Print layer information
        print("\nLayer Information:")
        for layer, nodes in self.layers.items():
            print(f"Layer {layer}: {len(nodes)} nodes")

        # Print network statistics
        depth = len(self.layers)
        max_breadth = max(len(layer) for layer in self.layers.values())
        print(f"\nNetwork Statistics:")
        print(f"Depth (number of layers): {depth}")
        print(f"Maximum Breadth (max nodes in a layer): {max_breadth}")

    def convert(self):
        self.build_graph()
        self.topological_sort()
        
        # Prepare neuron parameters
        neuron_params = []
        for node in self.genome.nodes:
            # You might want to evolve these parameters as well
            params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}
            neuron_params.append(params)
        
        # Get connections
        connections = self.get_connections()
        
        # Create and return the NEATSNN instance
        return NEATSNN(neuron_params, connections)

def create_network_from_genome(genome):
    neuron_params = {}
    for node in genome.nodes:
        neuron_params[node['node_id']] = {
            'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8,  # Default Izhikevich parameters
            'node_type': node['node_type']
        }
    
    connections = [
        {
            'in_node': conn['in_node'],
            'out_node': conn['out_node'],
            'weight': conn['weight']
        }
        for conn in genome.connections if conn['enabled']
    ]
    
    return NEATSNN(neuron_params, connections)