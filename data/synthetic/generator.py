import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def calculate_signal_strength(distance, signal_config):
    # Simple path loss model using config parameters
    base_signal = signal_config['base_strength']
    if distance == 0:
        return base_signal
    
    # Log-distance path loss model
    path_loss = signal_config['path_loss_exponent'] * np.log10(distance / 100)
    return base_signal - path_loss

def generate_synthetic_mobility_data(num_users, days, sampling_rate_seconds, grid_config, signal_config, patterns_config, handover_config):
    # Define cell tower locations (grid-based for simplicity)
    grid_size = grid_config['size']
    spacing = grid_config['spacing']
    coverage_radius = grid_config['coverage_radius']
    
    cell_towers = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell_towers.append({
                'id': f'cell_{i}_{j}',
                'x': i * spacing,
                'y': j * spacing,
                'coverage_radius': coverage_radius
            })
    
    # Create mobility patterns
    data = []
    start_time = datetime.now()
    
    for user_id in range(num_users):
        # Choose a movement pattern type for this user
        pattern_type = np.random.choice(['commuter', 'random_walk', 'stationary', 'high_mobility'])
        
        # Set starting position
        if pattern_type == 'commuter':
            # Commuters typically start at home locations
            home_area = patterns_config['commuter']['home_area']
            x = np.random.uniform(home_area['x_min'], home_area['x_max'])
            y = np.random.uniform(home_area['y_min'], home_area['y_max'])
        else:
            # Random starting position within the grid
            x = np.random.uniform(0, grid_size * spacing)
            y = np.random.uniform(0, grid_size * spacing)
        
        # Generate movement for each timestep
        current_time = start_time
        
        # Set initial velocity based on pattern type
        if pattern_type == 'random_walk':
            velocity = np.random.uniform(*patterns_config['random_walk']['velocity'])
        elif pattern_type == 'stationary':
            velocity = patterns_config['stationary']['velocity']
        elif pattern_type == 'high_mobility':
            velocity = np.random.uniform(*patterns_config['high_mobility']['velocity'])
        else:  # commuter
            velocity = np.random.uniform(*patterns_config['commuter']['velocities']['stationary'])
        
        heading = np.random.uniform(0, 2*np.pi)  # initial direction
        
        for day in range(days):
            for hour in range(24):
                # Adjust behavior based on time of day
                if pattern_type == 'commuter':
                    commute_hours = patterns_config['commuter']['commute_hours']
                    # Morning commute
                    if commute_hours['morning'][0] <= hour < commute_hours['morning'][1]:
                        # Head to work area
                        work_area = patterns_config['commuter']['work_area']
                        target_x = np.random.uniform(work_area['x_min'], work_area['x_max'])
                        target_y = np.random.uniform(work_area['y_min'], work_area['y_max'])
                        heading = np.arctan2(target_y - y, target_x - x)
                        velocity = np.random.uniform(*patterns_config['commuter']['velocities']['commute'])
                    # Evening commute
                    elif commute_hours['evening'][0] <= hour < commute_hours['evening'][1]:
                        # Head back home
                        home_area = patterns_config['commuter']['home_area']
                        target_x = np.random.uniform(home_area['x_min'], home_area['x_max'])
                        target_y = np.random.uniform(home_area['y_min'], home_area['y_max'])
                        heading = np.arctan2(target_y - y, target_x - x)
                        velocity = np.random.uniform(*patterns_config['commuter']['velocities']['commute'])
                    # At work or at home
                    else:
                        velocity = np.random.uniform(*patterns_config['commuter']['velocities']['stationary'])
                        # Random direction changes
                        if np.random.random() < 0.1:
                            heading = np.random.uniform(0, 2*np.pi)
                
                # Generate positions for this hour
                samples_per_hour = 3600 // sampling_rate_seconds
                for s in range(samples_per_hour):
                    # Add some randomness to movement
                    if np.random.random() < 0.2:
                        heading += np.random.uniform(-0.5, 0.5)
                    
                    # Update position
                    dx = velocity * sampling_rate_seconds * np.cos(heading)
                    dy = velocity * sampling_rate_seconds * np.sin(heading)
                    
                    x += dx
                    y += dy
                    
                    # Ensure within boundaries
                    x = max(0, min(x, grid_size * spacing))
                    y = max(0, min(y, grid_size * spacing))
                    
                    # Find closest cell tower
                    distances = [np.sqrt((x - tower['x'])**2 + (y - tower['y'])**2) for tower in cell_towers]
                    connected_cell_idx = np.argmin(distances)
                    connected_cell = cell_towers[connected_cell_idx]
                    signal_strength = calculate_signal_strength(distances[connected_cell_idx], signal_config)
                    
                    # Find potential handover candidates
                    handover_candidates = []
                    for idx, dist in enumerate(distances):
                        if idx != connected_cell_idx and dist < connected_cell['coverage_radius'] * 1.5:
                            candidate_strength = calculate_signal_strength(dist, signal_config)
                            handover_candidates.append({
                                'cell_id': cell_towers[idx]['id'],
                                'signal_strength': candidate_strength
                            })
                    
                    # Determine if handover is needed
                    handover_needed = False
                    handover_target = None

                    for candidate in handover_candidates:
                        signal_diff = candidate['signal_strength'] - signal_strength
                        
                        # Use appropriate threshold based on mobility pattern
                        base_threshold = handover_config['base_threshold']
                        if pattern_type == 'high_mobility':
                            base_threshold = handover_config['high_mobility_threshold']
                        
                        # Random chance of handover when signals are close
                        if signal_diff > base_threshold or (signal_diff > -1 and np.random.random() < handover_config['random_chance']):
                            handover_needed = True
                            handover_target = candidate['cell_id']
                            break
                        
                        # For users near cell boundaries, occasionally trigger handovers
                        edge_distance = abs(connected_cell['coverage_radius'] - distances[connected_cell_idx])
                        if edge_distance < handover_config['edge_distance_threshold'] and np.random.random() < handover_config['edge_handover_chance']:
                            handover_needed = True
                            handover_target = candidate['cell_id']
                            break
                    
                    # Record data point
                    timestamp = current_time + timedelta(days=day, hours=hour, seconds=s*sampling_rate_seconds)
                    data.append({
                        'timestamp': timestamp,
                        'user_id': f'user_{user_id}',
                        'x': x,
                        'y': y,
                        'velocity': velocity,
                        'heading': heading,
                        'connected_cell': connected_cell['id'],
                        'signal_strength': signal_strength,
                        'handover_needed': handover_needed,
                        'handover_target': handover_target,
                        'pattern_type': pattern_type
                    })
    
    return pd.DataFrame(data)

def visualize_trajectories(df, sample_users=5, grid_config=None):
    plt.figure(figsize=(12, 10))
    
    user_ids = df['user_id'].unique()
    selected_users = np.random.choice(user_ids, min(sample_users, len(user_ids)), replace=False)
    
    for user_id in selected_users:
        user_data = df[df['user_id'] == user_id].sort_values('timestamp')
        plt.plot(user_data['x'], user_data['y'], 'o-', alpha=0.7, label=f"{user_id} ({user_data['pattern_type'].iloc[0]})")
        
        # Mark handovers
        handovers = user_data[user_data['handover_needed'] == True]
        plt.scatter(handovers['x'], handovers['y'], color='red', s=50, marker='x')
    
    # Plot cell towers
    if grid_config:
        grid_size = grid_config['size']
        spacing = grid_config['spacing']
        
        for i in range(grid_size):
            for j in range(grid_size):
                plt.scatter(i*spacing, j*spacing, color='black', marker='s', s=100)
                plt.text(i*spacing, j*spacing+100, f'cell_{i}_{j}', fontsize=8)
    
    plt.title('Synthetic User Trajectories with Handover Points (red X)')
    plt.xlabel('X coordinate (meters)')
    plt.ylabel('Y coordinate (meters)')
    plt.legend()
    plt.grid(True)
    plt.show()
