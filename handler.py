from methods.evolutionary import evolutionary_controller
# from methods.learning import learning_controller

class Handler:
    def __init__(self, network_type, network_params, method_type, method_params, sim_params, environment=None):
        self.network_type = network_type
        self.network_params = network_params

        self.method_type = method_type
        self.method_params = method_params

        self.sim_params = sim_params

        self.environment = environment
        self._init_method()

    def _init_method(self):
        if self.method_type in ['NEAT']:
            controller = evolutionary_controller.Controller(self.network_type, self.network_params, self.method_type, self.method_params, self.sim_params, self.environment)
        elif self.method_type in ['q_learning', 'policy_gradient', 'actor_critic', 'soft_ac']:
            pass # This will call the learning_controller

