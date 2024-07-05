import numpy as np
import mlx.core as mx

from simulation.entity_config import EntityConfig
from simulation.body import Body
from simulation.receptors import collect_state

class Entity:
    def __init__(self):
        # Entity parameters
        self.config = EntityConfig()
        self.x = self.config.initial_x
        self.y = self.config.initial_y
        self.angle = self.config.initial_angle
        self.speed = self.config.speed
        self.turn_angle = np.radians(self.config.turn_angle)

        # Network parameters
        self.handler = None

        # This is a janky fix for the target being off by 500 in the y direction???!??!
        self.target = [self.config.target_x, self.config.target_y]
        self.target[1] -= 500

        # Create a Body
        self.body = Body()
        self.body.update(self.x, self.y, self.angle)

        # Starting delay
        self.delay = 300

        # Tracking for reward calculation
        self.highest_concentration = 0.0
        self.last_concentration = 0.0
        self.last_reward_position = None
        self.movement_threshold = 4.0
        self.static_count = 0
        self.time_step = 0
        self.time_penalty_factor = 0.01

    def update(self, environment):
        if self.delay > 0:
            self.delay -= 1
            return

        # Collect state from environment
        current_state = collect_state(self, environment)
        
        # Get action from handler
        action = self.handler.get_action(current_state)

        # Execute action
        self.execute_action(action)

        # Get new state and calculate reward
        new_state = collect_state(self, environment)
        reward = self.calculate_reward(new_state)

        # Update the handler
        self.handler.update(reward, new_state, done=False)

        self.time_step += 1

        return self.get_position()

    def execute_action(self, action):
        if action == 0:  # Turn left
            self.turn(-self.turn_angle)
        elif action == 1:  # Turn right
            self.turn(self.turn_angle)
        elif action == 2:  # Move forward
            self.move_forward(self.speed)
        else:  # Stay still
            print('Stay still')
            pass  # Do nothing

        self.body.update(self.x, self.y, self.angle)

    def turn(self, angle):
        self.angle += angle
        self.angle %= 2 * np.pi  # Keep angle between 0 and 2π

    def move_forward(self, distance):
        dx = np.cos(self.angle) * distance
        dy = np.sin(self.angle) * distance
        self.x += dx
        self.y += dy

    def get_position(self):
        return self.x, self.y, self.angle

    def get_state(self, environment):
        return collect_state(self, environment)

    def draw(self, screen):
        self.body.draw(screen, self.x, self.y, self.angle)

    def calculate_reward(self, state):
        # Get the maximum concentration from all receptors
        current_concentration = mx.max(state[-4:]).item()

        # Calculate current distance to target
        current_distance = self.calculate_distance_to_source()

        # Check if we've moved enough towards the target to consider a reward
        current_position = (self.x, self.y)
        if self.last_reward_position is None:
            self.last_reward_position = current_position
            self.highest_concentration = current_concentration
            return 0  # No reward on first calculation

        # Calculate movement towards the target
        last_distance_to_target = self.calculate_distance_to_source(self.last_reward_position)
        movement_towards_target = last_distance_to_target - current_distance

        # Initialize reward
        reward = 0

        # Reward for achieving new highest concentration
        if current_concentration > self.highest_concentration:
            concentration_improvement = current_concentration - self.highest_concentration
            reward += 50 * concentration_improvement  # Significant reward for new highest concentration
            self.highest_concentration = current_concentration

        # Reward for moving towards higher concentration (only if above the previous highest)
        if current_concentration > self.last_concentration and current_concentration >= self.highest_concentration:
            reward += 10 * (current_concentration - self.last_concentration)

        # Penalty for not moving enough
        if movement_towards_target < self.movement_threshold:
            self.static_count += 1
            reward -= 0.1 * self.static_count
        else:
            self.static_count = 0

        # Reward for moving towards the target (if concentration is increasing)
        if movement_towards_target > 0 and current_concentration > self.last_concentration:
            reward += 5 * movement_towards_target

        # Penalty for moving away from the target when concentration is zero
        if current_concentration == 0 and movement_towards_target < 0:
            reward -= 5

        # Apply time-based penalty
        time_penalty = self.time_penalty_factor * self.time_step
        reward -= time_penalty

        # Update tracking variables
        self.last_concentration = current_concentration
        self.last_reward_position = current_position

        return reward

    def calculate_distance_to_source(self, position=None):
        # Calculate Euclidean distance to the source
        if position is None:
            position = (self.x, self.y)
        dx = position[0] - self.target[0]
        dy = position[1] - self.target[1]
        return np.sqrt(dx**2 + dy**2)