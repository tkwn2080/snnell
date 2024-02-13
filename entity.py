import numpy as np
import pygame
from snn import Network





class OlfactoryEntity:
        def __init__(self, x, y, length, probe_angle, response_angle, distance, speed, network):
            self.x = x
            self.y = y
            self.speed = speed
            self.size = 12
            self.angle = 0  # Entity's facing angle, where 0 is facing to the right
            self.prong_length = length
            self.prong_thickness = 3
            self.prong_angle = probe_angle
            self.response_angle = response_angle

            self.L1_prong_x = None
            self.L1_prong_y = None
            self.R1_prong_x = None
            self.R1_prong_y = None
            self.L2_prong_x = None
            self.L2_prong_y = None
            self.R2_prong_x = None
            self.R2_prong_y = None

            self.prongs = [[self.L1_prong_x, self.L1_prong_y,], [self.R1_prong_x, self.R1_prong_y], [self.L2_prong_x, self.L2_prong_y], [self.R2_prong_x, self.R2_prong_y]]

            self.spikes = [0,0,0,0]
            
            # Status counters, etc.
            self.left_counter = 0
            self.right_counter = 0
            self.movement_counter = distance
            self.collided_with_emitter = False
            self.collision_time = None  # Time when collision occurred
            self.particle_count = 0
            self.timesteps_since_touch = 0
            
            # Tail wiggle attributes
            self.tail_wiggle_angle = np.pi / 12  # Maximum wiggle angle deviation
            self.tail_wiggle_speed = 0.2  # Speed of the wiggle
            self.wiggle_phase = 0  # Current phase of the wiggle

            self.network = network
            

        def draw(self, screen):

            # FIRST CIRCLE  
            pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), self.size)
            
            # SECOND CIRCLE
            recessed_x1 = self.x - np.cos(self.angle) * self.size * 0.75
            recessed_y1 = self.y - np.sin(self.angle) * self.size * 0.75
            pygame.draw.circle(screen, (0, 255, 0), (int(recessed_x1), int(recessed_y1)), self.size / 1.5)

            # THIRD CIRCLE
            recessed_x2 = recessed_x1 - np.cos(self.angle) * self.size * 0.75
            recessed_y2 = recessed_y1 - np.sin(self.angle) * self.size * 0.75
            pygame.draw.circle(screen, (0, 255, 0), (int(recessed_x2), int(recessed_y2)), self.size / 2)

            
            # FIRST PRONGS
            left_prong_x = self.x + np.cos(self.angle + self.prong_angle) * self.prong_length
            self.L1_prong_x = left_prong_x
            left_prong_y = self.y + np.sin(self.angle + self.prong_angle) * self.prong_length
            self.L1_prong_y = left_prong_y
            right_prong_x = self.x + np.cos(self.angle - self.prong_angle) * self.prong_length
            self.R1_prong_x = right_prong_x
            right_prong_y = self.y + np.sin(self.angle - self.prong_angle) * self.prong_length
            self.R1_prong_y = right_prong_y
            pygame.draw.line(screen, (0, 255, 0), (self.x, self.y), (left_prong_x, left_prong_y), 1)
            pygame.draw.line(screen, (0, 255, 0), (self.x, self.y), (right_prong_x, right_prong_y), 1)

            # SECOND PRONGS
            left_prong2_x = recessed_x1 + np.cos(self.angle + self.prong_angle) * self.prong_length
            self.L2_prong_x = left_prong2_x
            left_prong2_y = recessed_y1 + np.sin(self.angle + self.prong_angle) * self.prong_length
            self.L2_prong_y = left_prong2_y
            right_prong2_x = recessed_x1 + np.cos(self.angle - self.prong_angle) * self.prong_length
            self.R2_prong_x = right_prong2_x
            right_prong2_y = recessed_y1 + np.sin(self.angle - self.prong_angle) * self.prong_length
            self.R2_prong_y = right_prong2_y
            pygame.draw.line(screen, (0, 255, 0), (recessed_x1, recessed_y1), (left_prong2_x, left_prong2_y), 1)
            pygame.draw.line(screen, (0, 255, 0), (recessed_x1, recessed_y1), (right_prong2_x, right_prong2_y), 1)

            # TAIL
            wiggle_effect = np.sin(self.wiggle_phase) * self.tail_wiggle_angle
            tail_angle = self.angle + wiggle_effect
            tail_length = self.size
            tail_width = 2  
            tail_x = recessed_x2 - np.cos(tail_angle) * tail_length
            tail_y = recessed_y2 - np.sin(tail_angle) * tail_length
            pygame.draw.line(screen, (0, 255, 0), (recessed_x2, recessed_y2), (tail_x, tail_y), tail_width)

        def check_prong_collision(self, particle, prong_x, prong_y):
            prong_end_pos = np.array([prong_x, prong_y])
            entity_pos = np.array([self.x, self.y])
            particle_pos = np.array(particle[:2])  # Assuming particle format is [x, y, vx, vy]

            prong_vector = prong_end_pos - entity_pos
            particle_vector = particle_pos - entity_pos

            # Calculate the projection of the particle vector onto the prong vector
            projection = np.dot(particle_vector, prong_vector) / np.linalg.norm(prong_vector)
            projected_vector = projection * prong_vector / np.linalg.norm(prong_vector)
            closest_point = entity_pos + projected_vector

            # Check if the projection falls within the prong's length
            if np.linalg.norm(projected_vector) > np.linalg.norm(prong_vector) or projection < 0:
                return False

            # Calculate the distance from the closest point on the prong to the particle
            distance_to_particle = np.linalg.norm(closest_point - particle_pos)

            # Collision detected if the distance is less than the sum of the radii
            particle_radius = 2  # Assuming particle radius is 2
            return distance_to_particle <= (particle_radius + self.prong_thickness / 2)

        def move_response(self, distance, angle):
            # Updates the entity's position based on the distance and angle of movement
            self.x += np.cos(self.angle + angle) * distance
            self.y += np.sin(self.angle + angle) * distance


        # NEW WORKOUTPLAN
        def move(self, action):
            if np.array_equal(action, [1, 0, 0]):
                self.left_turn(self.response_angle)
            elif np.array_equal(action, [0, 1, 0]):
                self.straight_ahead(self.movement_counter)
            elif np.array_equal(action, [0, 0, 1]):
                self.right_turn(self.response_angle)

        def straight_ahead(self, distance):
            self.x += np.cos(self.angle) * distance
            self.y += np.sin(self.angle) * distance

        def left_turn(self, angle, distance):
            self.angle += angle
            self.x += np.cos(self.angle) * distance
            self.y += np.sin(self.angle) * distance

        def right_turn(self, angle, distance):
            self.angle -= angle
            self.x += np.cos(self.angle) * distance
            self.y += np.sin(self.angle) * distance

        def interpret_output(self, output):
            if output == [1, 0, 0]:
                self.left_turn(self.response_angle, self.movement_counter)
                self.wiggle_phase += self.tail_wiggle_speed
            elif output == [0, 1, 0]:
                self.straight_ahead(self.movement_counter)
                self.wiggle_phase += self.tail_wiggle_speed
            elif output == [0, 0, 1]:
                self.right_turn(self.response_angle, self.movement_counter)
                self.wiggle_phase += self.tail_wiggle_speed

        def handle_spikes(self, left_spike, right_spike):
            if left_spike and right_spike:
                spike = [1,1]
                self.particle_count += 2
            elif left_spike:
                spike = [1,0]
                self.particle_count +=1
            elif right_spike:
                spike = [0,1]
                self.particle_count += 1
            else:
                spike = [0,0]
            return spike    

        def update(self, red_particles):
            # Initialize spike indicators for this frame

            # Network.update_signal(self.network)

            for particle in red_particles:
                self.spikes = [0,0,0,0]
                
                for i in range(self.prongs):
                    if self.check_prong_collision(particle, self.prongs[i][0], self.prongs[i][1]):
                        self.spikes[i] = 1
                        self.particle_count += 1


            



            # # Handle movement based on counters for smooth transitions
            # if self.left_counter > 0 or self.right_counter > 0:
            #     # Calculate the difference and the sum of the counters
            #     counter_difference = self.left_counter - self.right_counter
            #     counter_sum = self.left_counter + self.right_counter
                
            #     # Normalize the sensitivity to the range [0, 1]
            #     normalized_sensitivity = max(0, min(self.sensitivity, 1))
                
            #     # Apply the normalized sensitivity to the counter difference
            #     adjusted_difference = counter_difference * normalized_sensitivity
                
            #     # Determine the direction and magnitude of the turn based on the adjusted counter difference
            #     turn_direction = -np.sign(adjusted_difference)  # Negative for left turn, positive for right turn
            #     turn_magnitude = abs(adjusted_difference) / counter_sum if counter_sum != 0 else 0
                
            #     # Move straight if counters are equal, otherwise turn proportionally to the imbalance
            #     if adjusted_difference == 0:
            #         self.move_response(self.speed, 0)  # Move straight
            #     else:
            #         self.move_response(self.speed, turn_direction * self.response_angle * turn_magnitude)
                
            #     # Decrease both counters
            #     self.left_counter = max(self.left_counter - 1, 0)
            #     self.right_counter = max(self.right_counter - 1, 0)

            #     # Increment the wiggle phase
            #     self.wiggle_phase += self.tail_wiggle_speed

            # Handle spikes
            spikes = self.handle_spikes (left_spike, right_spike)
            output = Network.propagate_spike(self.network, spikes, self.particle_count)
            self.interpret_output(output)