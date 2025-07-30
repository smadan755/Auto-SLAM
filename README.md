# Drone Visual Odometry Control System

This is a modular drone control system that combines visual odometry, real-time visualization, and keyboard input for drone navigation.

## File Structure

```
├── __init__.py                    # Package initialization
├── main.py                        # Main coordination logic
├── config.py                      # Configuration constants
├── input_handler.py               # Pygame keyboard input handling
├── gui_display.py                 # Real-time visualization (matplotlib + OpenCV)
├── drone_controller.py            # MAVSDK drone control
├── visual_odometry_processor.py   # Computer vision and VO processing
├── utils.py                       # Utility functions (CSV I/O, etc.)
└── VO_test copy.py               # Original monolithic file (for reference)
```

## Components

### 1. `config.py`
- Centralized configuration constants
- Easy to modify settings without touching core logic
- Includes AirSim IP, speeds, file paths, timing settings

### 2. `input_handler.py`
- Pygame-based keyboard input handling
- Runs in separate thread
- Maps keyboard inputs to velocity commands
- Handles landing command ('L' key)

### 3. `gui_display.py`
- Real-time visualization using matplotlib and OpenCV
- Displays drone camera feed with feature points
- Shows live path plot
- Runs in separate thread

### 4. `drone_controller.py`
- MAVSDK-based drone control
- Handles connection, arming, takeoff, landing
- Executes velocity commands from input handler
- Async function for non-blocking operation

### 5. `visual_odometry_processor.py`
- Processes AirSim camera feed
- Performs visual odometry calculations
- Annotates frames with feature points
- Tracks estimated path
- Async function for real-time processing

### 6. `utils.py`
- Helper functions for data I/O
- CSV saving and loading for path data
- Reusable utility functions

### 7. `main.py`
- Coordinates all components
- Manages shared state (queues, events, data)
- Orchestrates threads and async tasks
- Handles cleanup and error handling

## Usage

From the parent directory, run the system with:
```bash
python run_modular_system.py
```

Or run directly from within the package:
```bash
cd drone_vo_system
python -m drone_vo_system.main
```

Or from the original file:
```bash
python "VO_test copy.py"
```

## Controls

- **W/S**: Forward/Backward
- **A/D**: Left/Right
- **UP/DOWN**: Up/Down
- **Q/E**: Rotate left/right
- **L**: Land and quit
- **Q** (in GUI window): Quit

## Dependencies

- airsim
- cv2 (OpenCV)
- numpy
- matplotlib
- pygame
- mavsdk
- asyncio
- threading
- queue
- csv

## Benefits of Modular Structure

1. **Maintainability**: Each component has a single responsibility
2. **Testability**: Components can be unit tested independently
3. **Reusability**: Components can be used in other projects
4. **Collaboration**: Different team members can work on different components
5. **Configuration**: Easy to modify settings without touching core logic
6. **Debugging**: Easier to isolate and fix issues in specific components
