# This will handle the receptor mechanisms
# It will handle the deformation of the antennae
# It will handle the receptor-mediated odour response
# The latter will probably be done based on concentration
# We can encode concentration as a time-to-firing (bORN)
# How does this occur within a timestep?
# If we swap the simulation to fluid dynamics, time is an open question
# Discrete time is necessary for determining mathematical relations
from body import Body
import mlx.core as mx

class Antennae:
    def get_spikes(state, antennae, angle):
        spikes = mx.zeros(6, dtype=mx.float32)
        wind = state
        spikes[:4] = Antennae.convert_wind(wind, angle)

        # Get the head receptor spikes
        head_receptor_spike1, head_receptor_spike2 = Antennae.get_heading(wind, angle)

        # Assign the head receptor spikes to specific indices in the spikes array
        spikes[4] = head_receptor_spike1
        spikes[5] = head_receptor_spike2

        return spikes

    def convert_wind(wind, entity_angle):
        # Convert wind to membrane potentials
        antenna_angles = mx.array([55, -55, 125, -125]) * (mx.pi / 180)  # Convert angles to radians
        antenna_length = 12 * 5
        max_windspeed = 140  # Maximum windspeed
        deformation_coefficient = 0.01  # Coefficient for antenna deformation
        conversion_factor = 1.0  # Conversion factor from deformation to membrane potential

        # Extract wind direction and speed
        wind_direction = wind[0]
        wind_speed = wind[1]

        # Calculate the wind vector
        wind_vector = mx.array([mx.cos(wind_direction), mx.sin(wind_direction)]) * wind_speed

        # Calculate the antenna vectors
        antenna_vectors = mx.array([
            [mx.cos(entity_angle + antenna_angles[0]), mx.sin(entity_angle + antenna_angles[0])],
            [mx.cos(entity_angle + antenna_angles[1]), mx.sin(entity_angle + antenna_angles[1])],
            [mx.cos(entity_angle + antenna_angles[2]), mx.sin(entity_angle + antenna_angles[2])],
            [mx.cos(entity_angle + antenna_angles[3]), mx.sin(entity_angle + antenna_angles[3])]
        ])

        # Calculate the dot product between the wind vector and antenna vectors
        perpendicular_forces = mx.tensordot(wind_vector, antenna_vectors, axes=[[0], [1]])

        # Calculate the deformation of each antenna
        deformations = perpendicular_forces * deformation_coefficient

        # Convert deformations to membrane potentials
        membrane_potentials = deformations * conversion_factor

        # Perform relative normalization and scaling
        min_potential = mx.min(membrane_potentials)
        max_potential = mx.max(membrane_potentials)
        normalized_potentials = (membrane_potentials - min_potential) / (max_potential - min_potential)

        return normalized_potentials

    def get_heading(wind, entity_angle):
        # Extract wind direction and speed
        wind_direction = wind[0]
        wind_speed = wind[1]

        # Calculate the relative angle between the entity and the wind direction
        relative_angle = wind_direction - entity_angle

        # Ensure the relative angle is within the range [-pi, pi]
        relative_angle = (relative_angle + mx.pi) % (2 * mx.pi) - mx.pi

        # Calculate the angle of incidence (in degrees)
        angle_of_incidence = mx.abs(relative_angle) * 180 / mx.pi

        # Calculate the spike values for the two head receptors
        if angle_of_incidence <= 90:
            spike1 = angle_of_incidence / 90
            spike2 = 1 - spike1
        else:
            spike1 = 0
            spike2 = 0

        return spike1, spike2


class Cilia:
    def __init__(self):
        self.eval_counter = 0

    def get_spikes(state, cilia):
        spikes = [0,0,0,0,0,0,0,0]
        concentrations = []

        # This is probably a dumb method
        radius = 1

        for i in enumerate(cilia):
            concentration = Cilia.get_concentrations(state, i[1][0], i[1][1], radius)
            concentrations.append(concentration)

        concentrations = mx.array(concentrations)
        spikes = Cilia.convert_concentrations(concentrations)

        # Currently this is just returning membrane potentials, not spikes
        return spikes

    def get_concentrations(state, x, y, radius):
        # Calculate the concentration of puffs at the given position and radius
        times = state['time']
        x_puffs = state['x']
        y_puffs = state['y']
        radii = state['radius']
        concentration = state['concentration']

        active_mask = times >= 0
        numeric_mask = active_mask.astype(mx.float32)

        # Gimmicky fix, standardising
        y = y - 400

        # Calculate the distances between the given position and all active puffs
        dx = x_puffs - x
        dy = y_puffs - y
        center_distances = mx.sqrt(dx**2 + dy**2)
        distances = center_distances - radii * 100

        # Check if the puffs are overlapping with the given position
        overlap_mask = distances <= (radius + radii)

        # Calculate the concentration contribution of each overlapping puff
        concentrations = concentration * overlap_mask

        # Apply the numeric mask to filter out inactive puffs
        concentrations *= numeric_mask

        # Sum the concentrations of all overlapping puffs
        total_concentration = mx.sum(concentrations)
        return total_concentration

    def convert_concentrations(concentrations):
        # Convert concentrations to membrane potentials
        # The maximum concentration is 100, so we need to scale it to a range of 0 to 1
        max_concentration = mx.array(100)
        membrane_potentials = mx.divide(concentrations, max_concentration)
        # return membrane_potentials

        # Instead normalise by concentration at that point
        epsilon = 1e-8  # Small value to avoid division by zero
        total_concentration = mx.sum(concentrations) + epsilon
        relative_concentrations = mx.divide(concentrations, total_concentration)
        return relative_concentrations