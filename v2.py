import numpy as np
import mlx.core as mx
import threading
import time

class Layer:
    def __init__(self, n_in, n_out):
        self.weights = mx.array(np.random.uniform(-1, 1, (n_in, n_out)))
        self.mem_p = mx.zeros((n_out,), dtype=mx.float32)
        self.thresholds = mx.ones((n_out,), dtype=mx.float32)
        self.decay_rate = 0.8

    def forward(self, spikes_in):
        weighted_input = mx.matmul(spikes_in, self.weights)
        self.mem_p = mx.add(self.mem_p, weighted_input)
        spikes_out = mx.greater_equal(self.mem_p, self.thresholds)
        print(f'Spikes Out: {spikes_out}')
        self.mem_p = mx.where(spikes_out, mx.zeros_like(self.mem_p), self.mem_p)
        print(f'Membrane Potentials: {self.mem_p}')
        self.mem_p = mx.multiply(self.mem_p, self.decay_rate)
        return spikes_out

class Network:
    def __init__(self, dim):
        self.layers = [Layer(dim[i], dim[i+1]) for i in range(len(dim)-1)]

    def forward(self, spikes):
        for layer in self.layers:
            spikes = layer.forward(spikes)
        return spikes

class Clock:
    def __init__(self, network, clock_rate):
        self.network = network
        self.clock_rate = clock_rate  # Clock rate in seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run_clock)
        self.thread.daemon = True  # Ensures the thread exits when the main program does

    def start(self):
        self.thread.start()

    def run_clock(self):
        while not self.stop_event.is_set():
            start_time = time.time()

            # Example spikes input for demonstration; adjust as necessary for your use case
            spikes_in = mx.array([1, 1, 1, 1, 0, 0], dtype=mx.float32)  # Adjust the shape/content as needed
            output_spikes = self.network.forward(spikes_in)
            print("Output Spikes:", output_spikes)

            # Wait for the next tick
            time.sleep(max(0, self.clock_rate - (time.time() - start_time)))

    def stop(self):
        self.stop_event.set()
        self.thread.join()

# Example usage
if __name__ == "__main__":
    # Define your network dimensions and initialize it
    dim = [6, 100, 100, 1000, 100, 2]
    net = Network(dim)

    # Initialize the network clock with the network and desired clock rate (e.g., 0.5 seconds)
    net_clock = Clock(net, 0.5)
    
    # Start the clock
    net_clock.start()

    # Run for some time, then stop
    time.sleep(5)  # Adjust as needed
    net_clock.stop()


