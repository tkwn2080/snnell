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

        # Target location
        self.target = [self.config.target_x, self.config.target_y]

        # Create a Body
        self.body = Body()
        self.body.update(self.x, self.y, self.angle)

        # Starting delay
        self.delay = 300

        # Tracking for reward calculation
        self.closest_distance = float('inf')
        self.last_distance = float('inf')
        self.last_position = (self.x, self.y)
        self.last_concentration = 0
        self.highest_concentration = 0
        self.plume_following_score = 0
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
        reward = self.calculate_reward(new_state, environment)

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
            pass

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

    def calculate_reward(self, state, environment):
        current_position = (self.x, self.y)
        current_distance = self.calculate_distance_to_source()
        current_concentration = self.get_current_concentration(state)

        # Initialize reward
        reward = 0

        # Reward for getting closer to the source
        if current_distance < self.closest_distance:
            improvement = self.closest_distance - current_distance
            reward += 5 * improvement  # Significant reward for new closest approach
            self.closest_distance = current_distance

        # Reward for following the plume
        plume_reward = self.calculate_plume_following_reward(current_concentration, current_distance)
        reward += plume_reward

        # Penalty for not moving
        if current_position == self.last_position:
            self.static_count += 1
            reward -= 0.1 * self.static_count
        else:
            self.static_count = 0

        # Apply time-based penalty
        time_penalty = self.time_penalty_factor * self.time_step
        reward -= time_penalty

        # Update tracking variables
        self.last_distance = current_distance
        self.last_position = current_position
        self.last_concentration = current_concentration

        return np.clip(reward, -10, 20)

    def get_current_concentration(self, state):
        # Assuming the last 4 elements of the state are the cilia concentrations
        return mx.sum(state[-4:]).item()

    def calculate_plume_following_reward(self, current_concentration, current_distance):
        concentration_change = current_concentration - self.last_concentration
        distance_change = self.last_distance - current_distance

        if concentration_change > 0:
            # Reward for moving towards higher concentration
            self.plume_following_score += 1
            if current_concentration > self.highest_concentration:
                self.highest_concentration = current_concentration
                return 10  # Bonus for finding new highest concentration
            return 5
        elif concentration_change < 0 and distance_change > 0:
            # Small reward for moving towards source even if concentration decreases
            self.plume_following_score += 0.5
            return 2
        elif concentration_change < 0 and distance_change < 0:
            # Penalty for moving away from source and losing concentration
            self.plume_following_score -= 1
            return -5
        else:
            # No change or other scenarios
            return 0

    def calculate_distance_to_source(self):
        dx = self.x - self.target[0]
        dy = self.y - self.target[1]
        return np.sqrt(dx**2 + dy**2)