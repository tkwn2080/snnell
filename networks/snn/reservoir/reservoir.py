import mlx.core as mx

class Reservoir:
    def __init__(self, n_in, n_res, n_pred, n_act, recurrence, weights):
        self.n_in = n_in
        self.n_res = n_res
        self.recurrence = recurrence # This is not currently used, but will be necessary where the reservoir is sparse
        self.weights = weights

        # Initialise the input layer
        self.a_in = mx.full((self.n_in,), 0.02)
        self.b_in = mx.full((self.n_in,), 0.2)
        self.c_in = mx.full((self.n_in,), -65)
        self.d_in = mx.full((self.n_in,), 8)

        self.v_in = mx.full((self.n_in,), -65)
        self.u_in = mx.multiply(self.b_in, self.v_in)
        self.threshold_in = mx.full((self.n_in,), -30)

        # Initialise the reservoir layer
        self.a_res = mx.full((self.n_res,), 0.02)
        self.b_res = mx.full((self.n_res,), 0.2)
        self.c_res = mx.full((self.n_res,), -65)
        self.d_res = mx.full((self.n_res,), 8)

        self.v_res = mx.full((self.n_res,), -65)
        self.u_res = mx.multiply(self.b_res, self.v_res)
        self.threshold_res = mx.full((self.n_res,), -30)

        # Initialize output layer (action neurons)
        self.a_out = mx.full((n_act,), 0.02)
        self.b_out = mx.full((n_act,), 0.2)
        self.c_out = mx.full((n_act,), -65)
        self.d_out = mx.full((n_act,), 8)
        
        self.v_out = mx.full((n_act,), -65)
        self.u_out = mx.multiply(self.b_out, self.v_out)
        self.threshold_out = mx.full((n_act,), -30)
        
        # Initialize prediction neuron
        self.v_pred = mx.array([-65.0])
        
        self.n_act = n_act
        self.n_pred = n_pred

        # Initialise the spike history
        self.spike_history = []

    def input(self, I_in):
        # Process input layer
        self.v_in = mx.add(mx.add(mx.add(mx.multiply(0.04, mx.square(self.v_in)), mx.multiply(5, self.v_in)), 140), mx.subtract(I_in, self.u_in))
        self.u_in = mx.add(self.u_in, mx.multiply(self.a_in, mx.subtract(mx.multiply(self.b_in, self.v_in), self.u_in)))

        input_spikes = mx.greater_equal(self.v_in, self.threshold_in)

        self.v_in = mx.where(input_spikes, self.c_in, self.v_in)
        self.u_in = mx.where(input_spikes, mx.add(self.u_in, self.d_in), self.u_in)

        # Propagate input spikes to reservoir
        reservoir_input = mx.matmul(input_spikes, self.weights['input'])
        
        # Generate initial reservoir spikes
        reservoir_spikes = mx.greater_equal(reservoir_input, self.threshold_res)

        return reservoir_spikes

    def forward(self, reservoir_spikes):
        I_res = mx.tensordot(reservoir_spikes, self.weights['reservoir'], axes=1)

        self.v_res = mx.add(mx.add(mx.add(mx.multiply(0.04, mx.square(self.v_res)), mx.multiply(5, self.v_res)), 140), mx.subtract(I_res, self.u_res))
        self.u_res = mx.add(self.u_res, mx.multiply(self.a_res, mx.subtract(mx.multiply(self.b_res, self.v_res), self.u_res)))

        spikes = mx.greater_equal(self.v_res, self.threshold_res)

        self.v_res = mx.where(spikes, self.c_res, self.v_res)
        self.u_res = mx.where(spikes, mx.add(self.u_res, self.d_res), self.u_res)

        return spikes, self.v_res

    def update_output(self, reservoir_spikes, timestep):
        # Update output layer to determine spikes

        # Record spikes to the history for the present timestep
        pass

    def get_output(self):
        return self.spike_history


class Network:
    def __init__(self, n_in, n_res, n_pred, n_act, recurrence, steps):
        self.n_in = n_in
        self.n_res = n_res
        self.n_pred = n_pred
        self.n_act = n_act
        self.n_out = n_act + n_pred
        self.recurrence = recurrence
        self.steps = steps

        self.weights = self.init_weights()
        self.reservoir = Reservoir(self.n_in, self.n_res, self.n_pred, self.n_act, self.recurrence, self.weights)
        self.output = Output(self.n_act, self.n_pred)

    def init_weights(self):
        w_in = mx.random.uniform(-0.1, 0.1, (self.n_in, self.n_res))

        if self.recurrence == 'dense':
            w_res = mx.random.uniform(-1, 1, (self.n_res, self.n_res))
            
            spectral_radius = 0.9
            current_radius = estimate_spectral_radius(w_res)
            w_res = w_res * (spectral_radius / current_radius)
        
        elif self.recurrence == 'sparse':
            w_res = mx.random.uniform(-1, 1, (self.n_res, self.n_res))
            mask = mx.random.uniform(0, 1, (self.n_res, self.n_res)) < 0.1
            w_res = mx.where(mask, w_res, 0)
            
            spectral_radius = 0.9
            current_radius = estimate_spectral_radius(w_res)
            w_res = w_res * (spectral_radius / current_radius)
        
        else:
            raise ValueError(f"Unknown recurrence type: {self.recurrence}")

        w_out = mx.random.uniform(-0.1, 0.1, (self.n_res, self.n_out))

        return {
            'input': w_in,
            'reservoir': w_res,
            'output': w_out
        }

    def propagate(self, input_signal):
        initial_spikes = self.reservoir.input(input_signal)
        reservoir_spikes = initial_spikes
        action_spike_history = []
        pred_potential_history = []
        
        for _ in range(self.steps):
            reservoir_spikes, _ = self.reservoir.forward(reservoir_spikes)
            self.reservoir.output(reservoir_spikes)

        spike_history = self.reservoir.get_output
        return spike_history

    def reset(self):
        self.output.reset()

    def compute_weighted_spikes(self, spike_history, decay_factor=0.9):
        weights = mx.array([decay_factor**i for i in range(len(spike_history))])
        weighted_spikes = mx.sum(spike_history * weights[:, mx.newaxis], axis=0)
        return weighted_spikes

    def get_reservoir_state(self):
        return self.reservoir.v_res

    def get_spike_history(self):
        return self.spike_history

    def update_output_weights(self, new_weights):
        self.weights['output'] = new_weights


def estimate_spectral_radius(matrix, num_iterations=100):
    n = matrix.shape[0]
    v = mx.random.normal(shape=(n, 1))
    for _ in range(num_iterations):
        v = mx.matmul(matrix, v)
        v = v / mx.linalg.norm(v)
    return mx.linalg.norm(mx.matmul(matrix, v))