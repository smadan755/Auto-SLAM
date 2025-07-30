import csv
from .config import DEFAULT_CSV_OUTPUT


def save_path_to_csv(path_data, filename=DEFAULT_CSV_OUTPUT):
    """Saves the estimated path (a list of (x, z) tuples) to a CSV file."""
    if not path_data:
        print("CSV: Path data is empty. Nothing to save.")
        return
    
    print(f"CSV: Saving path with {len(path_data)} points to {filename}...")
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'z'])  # Write header
            writer.writerows(path_data)  # Write all data points
        print(f"CSV: Successfully saved path to {filename}.")
    except IOError as e:
        print(f"CSV: Error saving file: {e}")


def load_path_from_csv(filename=DEFAULT_CSV_OUTPUT):
    """Loads path data from a CSV file."""
    try:
        path_data = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    path_data.append((float(row[0]), float(row[1])))
        print(f"CSV: Successfully loaded {len(path_data)} points from {filename}.")
        return path_data
    except FileNotFoundError:
        print(f"CSV: File {filename} not found.")
        return []
    except Exception as e:
        print(f"CSV: Error loading file: {e}")
        return []
