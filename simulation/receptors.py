import mlx.core as mx

def collect_state(entity, environment):
    antennae_state = get_antennae_state(entity, environment)
    cilia_state = get_cilia_state(entity, environment)
    return mx.concatenate([antennae_state, cilia_state])

def get_antennae_state(entity, environment):
    wind = environment['wind']
    angle = entity.angle

    spikes = mx.zeros(6, dtype=mx.float32)
    spikes[:4] = convert_wind(wind, angle)
    head_receptor_spike1, head_receptor_spike2 = get_heading(wind, angle)
    spikes[4] = head_receptor_spike1
    spikes[5] = head_receptor_spike2

    return spikes

def convert_wind(wind, entity_angle):
    antenna_angles = mx.array([55, -55, 125, -125]) * (mx.pi / 180)
    deformation_coefficient = 0.01
    conversion_factor = 1.0

    wind_direction, wind_speed = wind.direction, wind.speed
    wind_vector = mx.array([mx.cos(wind_direction), mx.sin(wind_direction)]) * wind_speed

    antenna_vectors = mx.array([
        [mx.cos(entity_angle + angle), mx.sin(entity_angle + angle)]
        for angle in antenna_angles
    ])

    perpendicular_forces = mx.tensordot(wind_vector, antenna_vectors, axes=[[0], [1]])
    deformations = perpendicular_forces * deformation_coefficient
    membrane_potentials = deformations * conversion_factor

    min_potential = mx.min(membrane_potentials)
    max_potential = mx.max(membrane_potentials)
    normalized_potentials = (membrane_potentials - min_potential) / (max_potential - min_potential + 1e-8)

    return normalized_potentials

def get_heading(wind, entity_angle):
    wind_direction, wind_speed = wind.direction, wind.speed
    relative_angle = (wind_direction - entity_angle + mx.pi) % (2 * mx.pi) - mx.pi
    angle_of_incidence = mx.abs(relative_angle) * 180 / mx.pi

    if angle_of_incidence <= 90:
        spike1 = angle_of_incidence / 90
        spike2 = 1 - spike1
    else:
        spike1 = spike2 = 0

    return spike1, spike2

def get_cilia_state(entity, environment):
    puffs = environment['puffs']
    cilia_positions = entity.body.get_cilia_positions()
    
    concentrations = mx.array([
        get_concentration(puffs, x, y)
        for x, y in cilia_positions
    ])
    
    return convert_concentrations(concentrations)

def get_concentration(puffs, x, y):
    active_mask = puffs['time'] >= 0
    numeric_mask = active_mask.astype(mx.float32)

    dx = puffs['x'] - x
    dy = puffs['y'] - (y - 400)  # Adjusting y as in the original code
    center_distances = mx.sqrt(dx**2 + dy**2)
    distances = center_distances - puffs['radius'] * 100

    overlap_mask = distances <= puffs['radius']
    concentrations = puffs['concentration'] * overlap_mask * numeric_mask

    return mx.sum(concentrations)

def convert_concentrations(concentrations):
    epsilon = 1e-8
    total_concentration = mx.sum(concentrations) + epsilon
    return mx.divide(concentrations, total_concentration)