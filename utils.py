import csv
import json
import numpy as np

# Try relative imports first (when run as module), fall back to absolute imports
try:
    from .config import DEFAULT_CSV_OUTPUT, ENABLE_3D_MAPPING
except ImportError:
    # Fallback for direct execution
    from config import DEFAULT_CSV_OUTPUT, ENABLE_3D_MAPPING


def save_path_to_csv(path_data, filename=DEFAULT_CSV_OUTPUT):
    """Enhanced function to save path data (2D or 3D) to CSV/JSON files."""
    if not path_data:
        print("CSV: Path data is empty. Nothing to save.")
        return
    
    print(f"CSV: Saving path with {len(path_data)} points to {filename}...")
    
    # Check if we have 3D data
    is_3d_data = (ENABLE_3D_MAPPING and path_data and 
                  isinstance(path_data[0], dict) and 'position' in path_data[0])
    
    try:
        if is_3d_data:
            # Save 3D data as JSON for full fidelity
            json_filename = filename.replace('.csv', '_3d.json')
            with open(json_filename, 'w') as f:
                json.dump(path_data, f, indent=2, default=str)
            print(f"CSV: Successfully saved 3D path to {json_filename}.")
            
            # Also save simplified CSV for compatibility
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y', 'z', 'timestamp'])
                for point in path_data:
                    pos = point['position']
                    timestamp = point.get('timestamp', 0)
                    writer.writerow([pos[0], pos[1], pos[2], timestamp])
            print(f"CSV: Successfully saved simplified 3D path to {filename}.")
        else:
            # Legacy 2D saving
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'z'])  # Write header
                writer.writerows(path_data)  # Write all data points
            print(f"CSV: Successfully saved 2D path to {filename}.")
            
    except IOError as e:
        print(f"CSV: Error saving file: {e}")


def save_3d_map_data(map_points, trajectory_3d, filename="slam_map_3d.json"):
    """Save complete 3D SLAM map data including points and trajectory."""
    if not ENABLE_3D_MAPPING:
        print("3D mapping not enabled. Skipping 3D map save.")
        return
        
    try:
        map_data = {
            'map_points': [],
            'trajectory_3d': trajectory_3d,
            'metadata': {
                'num_map_points': len(map_points),
                'num_trajectory_points': len(trajectory_3d),
                'format_version': '1.0'
            }
        }
        
        # Convert map points to serializable format
        for point in map_points:
            if hasattr(point, 'pt'):
                map_data['map_points'].append({
                    'position': point.pt.tolist(),
                    'quality': getattr(point, 'quality', 1.0),
                    'observations': getattr(point, 'observations', 1),
                    'color': getattr(point, 'color', [128, 128, 128]).tolist()
                })
            else:
                # Handle numpy array points
                map_data['map_points'].append({
                    'position': point.tolist(),
                    'quality': 1.0,
                    'observations': 1,
                    'color': [128, 128, 128]
                })
        
        with open(filename, 'w') as f:
            json.dump(map_data, f, indent=2, default=str)
        print(f"3D Map: Successfully saved complete 3D map to {filename}.")
        
    except Exception as e:
        print(f"3D Map: Error saving 3D map: {e}")


def load_path_from_csv(filename=DEFAULT_CSV_OUTPUT):
    """Enhanced function to load path data (2D or 3D) from CSV/JSON files."""
    try:
        # Try to load 3D JSON data first
        json_filename = filename.replace('.csv', '_3d.json')
        try:
            with open(json_filename, 'r') as f:
                path_data_3d = json.load(f)
            print(f"CSV: Successfully loaded 3D path with {len(path_data_3d)} points from {json_filename}.")
            return path_data_3d
        except FileNotFoundError:
            pass  # Fall back to CSV
        
        # Load CSV data (2D or simplified 3D)
        path_data = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Read header
            
            for row in reader:
                if len(header) >= 4 and len(row) >= 4:  # 3D CSV format
                    path_data.append({
                        'position': [float(row[0]), float(row[1]), float(row[2])],
                        'timestamp': float(row[3]) if row[3] else 0
                    })
                elif len(row) >= 2:  # 2D format
                    path_data.append((float(row[0]), float(row[1])))
                    
        print(f"CSV: Successfully loaded {len(path_data)} points from {filename}.")
        return path_data
        
    except FileNotFoundError:
        print(f"CSV: File {filename} not found.")
        return []
    except Exception as e:
        print(f"CSV: Error loading file: {e}")
        return []


def load_3d_map_data(filename="slam_map_3d.json"):
    """Load complete 3D SLAM map data including points and trajectory."""
    try:
        with open(filename, 'r') as f:
            map_data = json.load(f)
        
        print(f"3D Map: Successfully loaded 3D map from {filename}.")
        print(f"  Map Points: {map_data['metadata']['num_map_points']}")
        print(f"  Trajectory Points: {map_data['metadata']['num_trajectory_points']}")
        
        return map_data
        
    except FileNotFoundError:
        print(f"3D Map: File {filename} not found.")
        return None
    except Exception as e:
        print(f"3D Map: Error loading 3D map: {e}")
        return None


def analyze_3d_trajectory(trajectory_3d):
    """Analyze 3D trajectory for statistics and quality metrics."""
    if not trajectory_3d:
        return {}
    
    positions = np.array([t['position'] for t in trajectory_3d])
    
    # Calculate trajectory statistics
    total_distance = 0
    max_speed = 0
    
    for i in range(1, len(positions)):
        segment_distance = np.linalg.norm(positions[i] - positions[i-1])
        total_distance += segment_distance
        
        # Calculate speed if timestamps available
        if 'timestamp' in trajectory_3d[i] and 'timestamp' in trajectory_3d[i-1]:
            dt = trajectory_3d[i]['timestamp'] - trajectory_3d[i-1]['timestamp']
            if dt > 0:
                speed = segment_distance / dt
                max_speed = max(max_speed, speed)
    
    # Bounding box
    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions, axis=0)
    
    stats = {
        'total_points': len(trajectory_3d),
        'total_distance': total_distance,
        'max_speed': max_speed,
        'bounding_box': {
            'min': min_pos.tolist(),
            'max': max_pos.tolist(),
            'size': (max_pos - min_pos).tolist()
        },
        'altitude_range': [min_pos[1], max_pos[1]],
        'start_position': positions[0].tolist(),
        'end_position': positions[-1].tolist()
    }
    
    return stats
