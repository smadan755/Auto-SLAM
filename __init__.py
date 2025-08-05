"""
Drone Visual Odometry Control System

A modular system for controlling drones with visual odometry capabilities.

Components:
- config: Configuration constants
- input_handler: Pygame-based input handling
- gui_display: Real-time visualization
- drone_controller: MAVSDK drone control
- visual_odometry_processor: Computer vision processing
- utils: Utility functions for data I/O
- main: Main coordination logic
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main components for easy access
from main import main
from config import *
from utils import save_path_to_csv, load_path_from_csv

__all__ = [
    'main',
    'save_path_to_csv', 
    'load_path_from_csv'
]
