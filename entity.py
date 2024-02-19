import numpy as np
import pygame
from snn import Network

class OlfactoryEntity:
        def __init__(self, x, y, length, probe_angle, response_angle, distance, speed, network):
            self.x = x
            self.y = y
            self.speed = speed
            self.size = 12
            self.angle = 0
            # self.angle = np.random.uniform(0, 2*np.pi)  # Entity's facing angle, where 0 is facing to the right and the angle is randomly initialized
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
            self.update_prong_positions()

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

            self.closest_distance = 800
            self.last_five_distances = []
            
        def update_prong_positions(self):
            # Calculate and update the positions of all prongs based on the current entity position and angle
            self.L1_prong_x = self.x + np.cos(self.angle + self.prong_angle) * self.prong_length
            self.L1_prong_y = self.y + np.sin(self.angle + self.prong_angle) * self.prong_length
            self.R1_prong_x = self.x + np.cos(self.angle - self.prong_angle) * self.prong_length
            self.R1_prong_y = self.y + np.sin(self.angle - self.prong_angle) * self.prong_length

            recessed_x1 = self.x - np.cos(self.angle) * self.size * 0.75
            recessed_y1 = self.y - np.sin(self.angle) * self.size * 0.75

            self.L2_prong_x = recessed_x1 + np.cos(self.angle + self.prong_angle) * self.prong_length
            self.L2_prong_y = recessed_y1 + np.sin(self.angle + self.prong_angle) * self.prong_length
            self.R2_prong_x = recessed_x1 + np.cos(self.angle - self.prong_angle) * self.prong_length
            self.R2_prong_y = recessed_y1 + np.sin(self.angle - self.prong_angle) * self.prong_length

            # Update prongs list
            self.prongs = [[self.L1_prong_x, self.L1_prong_y], [self.R1_prong_x, self.R1_prong_y],
                        [self.L2_prong_x, self.L2_prong_y], [self.R2_prong_x, self.R2_prong_y]]

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
            forward_shift = self.size * 0.5
            left_prong_x = self.x + np.cos(self.angle + self.prong_angle) * self.prong_length + np.cos(self.angle) * forward_shift
            self.L1_prong_x = left_prong_x
            left_prong_y = self.y + np.sin(self.angle + self.prong_angle) * self.prong_length + np.sin(self.angle) * forward_shift
            self.L1_prong_y = left_prong_y
            right_prong_x = self.x + np.cos(self.angle - self.prong_angle) * self.prong_length + np.cos(self.angle) * forward_shift
            self.R1_prong_x = right_prong_x
            right_prong_y = self.y + np.sin(self.angle - self.prong_angle) * self.prong_length + np.sin(self.angle) * forward_shift
            self.R1_prong_y = right_prong_y
            pygame.draw.line(screen, (0, 255, 0), (self.x, self.y), (left_prong_x, left_prong_y), 1)
            pygame.draw.line(screen, (0, 255, 0), (self.x, self.y), (right_prong_x, right_prong_y), 1)

            # SECOND PRONGS
            backward_shift = self.size * 0.5
            left_prong2_x = recessed_x1 + np.cos(self.angle + self.prong_angle) * self.prong_length - np.cos(self.angle) * backward_shift
            self.L2_prong_x = left_prong2_x
            left_prong2_y = recessed_y1 + np.sin(self.angle + self.prong_angle) * self.prong_length - np.sin(self.angle) * backward_shift
            self.L2_prong_y = left_prong2_y
            right_prong2_x = recessed_x1 + np.cos(self.angle - self.prong_angle) * self.prong_length - np.cos(self.angle) * backward_shift
            self.R2_prong_x = right_prong2_x
            right_prong2_y = recessed_y1 + np.sin(self.angle - self.prong_angle) * self.prong_length - np.sin(self.angle) * backward_shift
            self.R2_prong_y = right_prong2_y
            pygame.draw.line(screen, (0, 255, 0), (recessed_x1, recessed_y1), (left_prong2_x, left_prong2_y), 1)
            pygame.draw.line(screen, (0, 255, 0), (recessed_x1, recessed_y1), (right_prong2_x, right_prong2_y), 1)
    
    
            # TAIL
            wiggle_effect = np.sin(self.wiggle_phase) * self.tail_wiggle_angle
            tail_angle = self.angle + wiggle_effect
            tail_length = self.size * 2
            tail_width = 3  
            tail_x = recessed_x2 - np.cos(tail_angle) * tail_length
            tail_y = recessed_y2 - np.sin(tail_angle) * tail_length
            pygame.draw.line(screen, (0, 255, 0), (recessed_x2, recessed_y2), (tail_x, tail_y), tail_width)

        def check_prong_collision(self, particle, prong_x, prong_y):

            if prong_x is None or prong_y is None:
                print("Prong position is None, skipping collision check")  # Temporary debugging aid
                return False

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
        
        def check_emitter_collision(self, emitter, prong_x, prong_y):
            if prong_x is None or prong_y is None:
                print("Prong position is None, skipping collision check")
                return False
            
            prong_end_pos = np.array([prong_x, prong_y])
            emitter_pos = np.array(emitter[:2])
            entity_pos = np.array([self.x, self.y])

            prong_vector = prong_end_pos - entity_pos
            emitter_vector = emitter_pos - entity_pos

            # Calculate the projection of the emitter vector onto the prong vector
            projection = np.dot(emitter_vector, prong_vector) / np.linalg.norm(prong_vector)
            projected_vector = projection * prong_vector / np.linalg.norm(prong_vector)
            closest_point = entity_pos + projected_vector

            # Check if the projection falls within the prong's length
            if np.linalg.norm(projected_vector) > np.linalg.norm(prong_vector) or projection < 0:
                return False
            
            # Calculate the distance from the closest point on the prong to the emitter
            distance_to_emitter = np.linalg.norm(closest_point - emitter_pos)

            # Collision detected if the distance is less than the sum of the radii
            emitter_radius = 10  # Assuming emitter radius is 10
            return distance_to_emitter <= (emitter_radius + self.prong_thickness / 2)
        
        def homing_radar(self, emitter):
            # Calculate the distance between the emitter and the entity
            current_distance = np.linalg.norm(np.array([self.x, self.y]) - np.array(emitter[:2]))
            
            # Store the current distance in the list of last 5 distances
            self.last_five_distances.append(current_distance)
            
            # If we have more than 5 distances, remove the oldest one
            if len(self.last_five_distances) > 20:
                self.last_five_distances.pop(0)
            
            # Calculate velocities based on the distances
            if len(self.last_five_distances) > 1:
                velocities = [self.last_five_distances[i+1] - self.last_five_distances[i] for i in range(len(self.last_five_distances)-1)]
            else:
                velocities = [0]
            
            # Calculate acceleration based on the velocities
            if len(velocities) > 1:
                accelerations = [velocities[i+1] - velocities[i] for i in range(len(velocities)-1)]
            else:
                accelerations = [0]
            
            # Check if the entity is accelerating towards the emitter based on the last 5 distances
            # and if the current distance is the smallest so far
            if len(accelerations) > 0 and accelerations[-1] > 0 and current_distance == min(self.last_five_distances):
                # print(f'Entity is accelerating towards the emitter: {current_distance}')
                return 1
            else:
                return 0
            

        def move_response(self, distance, angle):
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
            if distance < 0:
                print("Distance cannot be negative, switching")
            distance = abs(distance)
            self.x += np.cos(self.angle) * distance
            self.y += np.sin(self.angle) * distance

        def left_turn(self, angle, distance):
            self.angle += angle
            if distance < 0:
                print("Distance cannot be negative, switching")
            distance = abs(distance)
            self.x += np.cos(self.angle) * distance
            self.y += np.sin(self.angle) * distance

        def right_turn(self, angle, distance):
            self.angle -= angle
            if distance < 0:
                print("Distance cannot be negative, switching")
            distance = abs(distance)
            self.x += np.cos(self.angle) * distance
            self.y += np.sin(self.angle) * distance

        def interpret_output(self, output):
            if output[0] == 1:
                self.left_turn(self.response_angle, self.movement_counter)
                self.wiggle_phase += self.tail_wiggle_speed
            elif output[0] > 0:
                self.left_turn(self.response_angle * output[0], self.movement_counter * output[0])
                self.wiggle_phase += self.tail_wiggle_speed * output[0]
            if (output[1] == 1) and (output[2] == 1):
                # print('D-D-D-DOUBLE SPEEEED')
                self.straight_ahead(self.movement_counter)
                self.wiggle_phase += self.tail_wiggle_speed
                self.straight_ahead(self.movement_counter)
                self.wiggle_phase += self.tail_wiggle_speed
            elif (output[1] == 1) or (output[2] == 1):
                self.straight_ahead(self.movement_counter)
                self.wiggle_phase += self.tail_wiggle_speed
            elif (output[1] > 0):
                self.straight_ahead(self.movement_counter * output[1])
                self.wiggle_phase += self.tail_wiggle_speed * (output[1])
            elif (output[2] > 0):
                self.straight_ahead(self.movement_counter * output[2])
                self.wiggle_phase += self.tail_wiggle_speed * (output[2])
                
            if (output[1] == 1) and (output[2] == 1):
                self.straight_ahead(self.movement_counter)
                self.wiggle_phase += self.tail_wiggle_speed
                self.straight_ahead(self.movement_counter)
                self.wiggle_phase += self.tail_wiggle_speed
            if output[3] == 1:
                self.right_turn(self.response_angle, self.movement_counter)
                self.wiggle_phase += self.tail_wiggle_speed
            elif output[3] > 0:
                self.right_turn(self.response_angle * output[3], self.movement_counter * output[3])
                self.wiggle_phase += self.tail_wiggle_speed * output[3]

            # TEST: MEMBRANE POTENTIAL, PERCENTAGES
            

        def update(self, red_particles, emitter):

            # reward_signal = self.homing_radar(emitter)

            # if reward_signal == 1:
            #     learning_signal = 10
            #     Network.modify_learning(self.network, learning_signal)
            # elif reward_signal == 0:
            #     learning_signal = -10
            #     Network.modify_learning(self.network, learning_signal)
            #     # print('BEEPBEEPBEP')
            
            self.spikes = [0,0,0,0,0,0]

            for particle in red_particles:
                
                for i, prong in enumerate(self.prongs):
                    if self.check_prong_collision(particle, prong[0], prong[1]):
                        self.spikes[i] = 1
                        self.particle_count += 1

            # Check emitter collision for left prongs
            if self.check_emitter_collision(emitter, self.prongs[0][0], self.prongs[0][1]) or self.check_emitter_collision(emitter, self.prongs[1][0], self.prongs[1][1]):
                self.spikes[4] = 1

            # Check emitter collision for right prongs
            if self.check_emitter_collision(emitter, self.prongs[2][0], self.prongs[2][1]) or self.check_emitter_collision(emitter, self.prongs[3][0], self.prongs[3][1]):
                self.spikes[5] = 1

            # print(self.spikes)
            output = Network.propagate_spike(self.network, self.spikes, self.particle_count)
            self.interpret_output(output)