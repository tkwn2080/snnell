# This will draw the entity
# The entity should have plastic antennae with olfactory ciliae
# How do we encode at the receptor level?
from receptors import Antennae, Cilia
from snn import Network
import numpy as np
from body import Body
import mlx.core as mx

class Entity:
    def __init__(self, network, xy):
        self.x = xy[0]
        self.y = xy[1]
        self.angle = 0
        self.speed = 1
        self.response_angle = 1 * (np.pi / 180)

        self.body = Body()
        self.input_potential = mx.zeros(12, dtype=mx.float32)
        self.spikes = mx.zeros(12, dtype=mx.float32)
        self.thresholds = mx.ones_like(self.spikes)
        self.counter = 250

        # Status counters, etc.
        self.movement_counter = 1
        self.collided_with_emitter = False
        self.collision_time = None
        self.particle_count = 0
        self.timesteps_since_touch = 0

        # SNN
        self.network = network

    def get_entity(self):
        return self.x, self.y, self.angle

    def draw(self, screen):
        self.body.draw(screen, self.x, self.y, self.angle)

    def straight_ahead(self, distance):
        dx = np.cos(self.angle) * distance
        dy = np.sin(self.angle) * distance
        self.x += dx
        self.y += dy
        Body.wiggle(self.body, distance)
        # print(f'Straight ahead by {distance}')

    def left_turn(self, angle):
        self.angle += angle
        # print(f'Left by {angle} degrees')

    def right_turn(self, angle):
        self.angle -= angle
        # print(f'Right by {angle} degrees')

    def interpret_output(self, output):
        # if output[0] > 0:
        #     turn_angle = self.response_angle * output[0]
        #     self.left_turn(turn_angle)
        #     print(f'Turn left by {turn_angle}')

        # forward_distance = self.movement_counter * ((output[1] + output[2]) / 2)
        # self.straight_ahead(forward_distance)

        # if output[3] > 0:
        #     turn_angle = self.response_angle * output[3]
        #     self.right_turn(turn_angle)

        # Select the appropriate action based on the maximum output value
        values = mx.softmax(output)
        diffs = mx.abs(values - values[0])
        are_equal = mx.max(diffs) < 1e-6
        if are_equal:
            # print('beep')
            pass
        else:
            action = mx.argmax(output)
            if action == 0:
                self.left_turn(self.response_angle)
            elif action == 1:
                self.straight_ahead(self.movement_counter)
            elif action == 2:
                self.straight_ahead(self.movement_counter)
            elif action == 3:
                self.right_turn(self.response_angle)

    def update(self, state):
        self.body.update(self.x, self.y, self.angle)
        if self.counter == 0:
            spikes = mx.array(self.spikes)
            input_current = []

            antennae = self.body.get_antennae(self.x, self.y, self.angle)
            antennae_spikes = Antennae.get_spikes(state[0], antennae, self.angle)

            cilia = self.body.get_cilia(self.x, self.y)
            cilia_spikes = Cilia.get_spikes(state[1], cilia)

            input_current.extend(antennae_spikes)
            input_current.extend(cilia_spikes)

            input_current = mx.array(input_current)
            
            output = Network.inject_current(self.network, input_current)
            # print(f'Output: {output}')
            self.interpret_output(output)

        # elif self.counter == 10:
        #     self.counter -= 1
        #     # self.angle = mx.random.uniform(-30, 30) * mx.pi / 180
        #     # print(f"Entity angle: {self.angle}")

        else:
            self.counter -= 1
        