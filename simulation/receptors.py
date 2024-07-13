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

    wind_vector = mx.array([mx.cos(wind.direction), mx.sin(wind.direction)]) * wind.speed

    antenna_vectors = mx.array([
        [mx.cos(entity_angle + angle), mx.sin(entity_angle + angle)]
        for angle in antenna_angles
    ])

    perpendicular_forces = mx.sum(wind_vector * antenna_vectors, axis=1)

    deformations = perpendicular_forces * 0.01
    membrane_potentials = deformations * 1.0

    min_potential = mx.min(membrane_potentials)
    max_potential = mx.max(membrane_potentials)
    normalized_potentials = (membrane_potentials - min_potential) / (max_potential - min_potential + 1e-8)

    return normalized_potentials

def get_heading(wind, entity_angle):
    wind_direction, wind_speed = wind.direction, wind.speed
    relative_angle = (wind_direction - entity_angle + mx.pi) % (2 * mx.pi) - mx.pi
    
    # If relative_angle is between -pi/2 and pi/2, wind is coming from the front
    # Otherwise, it's coming from the back
    is_front = mx.abs(relative_angle) <= mx.pi/2
    
    front_force = mx.where(is_front, mx.cos(relative_angle), 0)
    back_force = mx.where(is_front, 0, -mx.cos(relative_angle))
    
    return front_force, back_force

def get_concentration(puffs, x, y):
    active_mask = puffs['time'] >= 0
    numeric_mask = active_mask.astype(mx.float32)

    dx = puffs['x'] - x
    dy = puffs['y'] - (y - 500)  # Adjust y to match the puff coordinate system
    distances = mx.sqrt(dx**2 + dy**2)

    # Use the same radius scaling as in drawing
    scaled_radius = puffs['radius'] * 100

    within_radius = distances <= scaled_radius
    
    # Get the maximum concentration from puffs within radius
    concentrations = mx.where(within_radius, puffs['concentration'], 0) * numeric_mask
    max_concentration = mx.max(concentrations)

    return max_concentration

def convert_concentrations(concentrations, initial_concentration):
    # Normalize concentrations based on initial concentration
    normalized_concentrations = concentrations / initial_concentration
    
    # Clip values to ensure they're between 0 and 1
    return mx.clip(normalized_concentrations, 0, 1)

def get_cilia_state(entity, environment):
    puffs = environment['puffs']
    cilia_positions = entity.body.get_cilia_positions()
    initial_concentration = 10000
    
    concentrations = mx.array([
        get_concentration(puffs, x, y)
        for x, y in cilia_positions
    ])
    
    normalized_concentrations = convert_concentrations(concentrations, initial_concentration)
    
    return normalized_concentrations