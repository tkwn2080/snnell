import numpy as np
import mlx.core as mx

from simulation.entity_config import EntityConfig
from simulation.body import Body
from simulation.receptors import collect_state

class Entity:
    def __init__(self, handler, network, trial_number):
        self.trial_number = trial_number
        self.config = EntityConfig(self.trial_number)
        self.x = self.config.initial_x
        self.y = self.config.initial_y
        self.angle = self.config.initial_angle
        self.speed = self.config.speed
        self.turn_angle = self.config.turn_angle
        self.max_speed = self.config.max_speed
        self.max_turn_rate = np.radians(self.config.turn_angle)
        
        self.handler = handler
        self.network = network
        self.action_type = self.handler.action_type
        
        self.target = [self.config.target_x, self.config.target_y - 500]  # Adjusted for y-offset
        
        self.body = Body()
        self.body.update(self.x, self.y, self.angle)
        
        self.delay = self.config.delay
        self.previous_distance = self.calculate_distance_to_source()
        self.previous_angle = self.angle

        self.concentration = []

    def e_update(self, environment):
        if self.delay > 0:
            self.delay -= 1
            return
        
        current_state = collect_state(self, environment)

        self.concentration = current_state[-4:] # Collect concentration for fitness calculation

        input_current = self.handler.input_encoding(current_state)
        action = self.handler.get_action(self.network, input_current)
        self.execute_action(action)
        return self.get_position()

    def l_update(self, environment):
        if self.delay > 0:
            self.delay -= 1
            return

        stored_environment = environment

        current_state = collect_state(self, environment)

        action = self.handler.get_action(current_state)

        self.execute_action(action)

        new_state = collect_state(self, stored_environment)

        reward = self.calculate_reward(new_state)

        self.handler.update(reward, new_state, done=False)

        return self.get_position()

    def execute_action(self, action):
        # print(f'Action: {action}')
        if self.action_type == 'discrete':
            if action == 0:  # Turn left
                self.turn(-self.max_turn_rate)
            elif action == 1:  # Turn right
                self.turn(self.max_turn_rate)
            elif action == 2:  # Move forward
                self.move_forward(self.speed)
        elif self.action_type == 'continuous':
            if self.network.network_type == 'spiking':
                left, straight, right = action
                turn = np.clip(left - right, -self.max_turn_rate, self.max_turn_rate)
                self.turn(turn * self.turn_angle)
                self.move_forward(straight * self.speed)
            else:
                movement, rotation = action

                # Bound the movement to prevent backwards motion
                movement = max(0, min(movement, 1))  # Clamp between 0 and 1
                
                # Limit the rotation to prevent sharp turns
                max_rotation = np.pi / 4  # 45 degrees
                rotation = max(-max_rotation, min(rotation, max_rotation))

                self.move_forward(movement * self.max_speed)
                self.turn(rotation)

        self.body.update(self.x, self.y, self.angle)

    def move_forward(self, distance):
        self.x += np.cos(self.angle) * distance
        self.y += np.sin(self.angle) * distance

    def turn(self, angle):
        self.angle = (self.angle + angle) % (2 * np.pi)

    def get_state(self, environment):
        return collect_state(self, environment)

    def get_position(self):
        return self.x, self.y, self.angle

    def get_concentration(self):
        return self.concentration

    def draw(self, screen):
        self.body.draw(screen, self.x, self.y, self.angle)

    def calculate_reward(self, state):
        current_position = (self.x, self.y)
        current_concentration = mx.max(state[-4:]).item()
        current_distance = self.calculate_distance_to_source(current_position)
        
        progress = self.previous_distance - current_distance
        self.previous_distance = current_distance

        rotation = abs(self.angle - self.previous_angle)
        self.previous_angle = self.angle

        reward = 0

        if current_distance < self.config.target_threshold:
            reward += 100
        
        reward += progress * 10
        reward += current_concentration * 5

        if rotation > np.pi/4:
            reward -= 1

        if not (0 <= self.x <= self.config.environment_width and 0 <= self.y <= self.config.environment_height):
            reward -= 5

        return reward

    def calculate_distance_to_source(self, position=None):
        if position is None:
            position = (self.x, self.y)
        dx = position[0] - self.target[0]
        dy = position[1] - self.target[1]
        return np.sqrt(dx**2 + dy**2)