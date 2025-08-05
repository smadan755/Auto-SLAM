# Configuration constants for the drone control system

# AirSim settings
AIRSIM_IP = "172.30.144.1"

# MAVSDK settings
DRONE_CONNECTION = "udp://:14540"

# Control settings
DEFAULT_SPEED = 5.0
YAW_SPEED_RATE = 60.0

# GUI settings
FIGURE_SIZE = (8, 8)
GUI_QUEUE_SIZE = 1

# File paths
CALIBRATION_FILE = "calib.txt"
DEFAULT_CSV_OUTPUT = "vo_path.csv"

# Timing settings - optimized for performance
ASYNC_SLEEP_INTERVAL = 0.02  # Increased to reduce CPU usage (50 FPS max)
PYGAME_WAIT_TIME = 10
GUI_TIMEOUT = 0.1
TAKEOFF_WAIT_TIME = 5
CONTROL_LOOP_INTERVAL = 0.05  # Increased control frequency for better responsiveness
FRAME_SKIP_COUNT = 2  # Process every 2nd frame for better performance

# Improved VO settings - performance optimized
MOTION_DETECTION_THRESHOLD = 2.0  # Lower threshold for better motion detection
MIN_FEATURES_FOR_VO = 25  # Reduced minimum features for faster processing
TEMPORAL_SMOOTHING_ALPHA = 0.7  # Pose smoothing factor (0.7 = 70% new, 30% old)
MAX_POSE_HISTORY = 3  # Reduced history for less memory usage

# Feature detector settings - performance optimized
SIFT_FEATURES = 300  # Reduced from 1000 for much faster detection
SIFT_CONTRAST_THRESHOLD = 0.04  # Slightly higher for fewer but better features
SIFT_EDGE_THRESHOLD = 10  # Edge response threshold

# ORB detector settings - faster alternative to SIFT
USE_ORB_DETECTOR = True  # Use ORB instead of SIFT for speed
ORB_FEATURES = 500  # Maximum ORB features to detect
ORB_SCALE_FACTOR = 1.2  # Pyramid decimation ratio
ORB_N_LEVELS = 8  # Number of pyramid levels

# Feature matching settings - performance optimized
RATIO_TEST_THRESHOLD = 0.75  # Slightly relaxed for more matches
RANSAC_THRESHOLD = 1.0  # RANSAC inlier threshold (pixels)
RANSAC_CONFIDENCE = 0.99  # RANSAC confidence level
USE_FLANN_MATCHER = True  # Use FLANN for faster matching

# Performance optimization settings
ENABLE_MULTITHREADING = True  # Enable parallel processing where possible
MAX_TRIANGULATION_POINTS = 50  # Limit triangulation for speed

# 3D Mapping settings
ENABLE_3D_MAPPING = True  # Enable full 3D map storage and visualization
MAX_MAP_POINTS = 1000  # Maximum number of 3D points to keep in memory
MAP_POINT_QUALITY_THRESHOLD = 0.01  # Minimum quality for map points
ENABLE_3D_VISUALIZATION = True  # Enable 3D visualization with matplotlib
MAP_UPDATE_FREQUENCY = 5  # Update 3D map every N frames
TRAJECTORY_HISTORY_SIZE = 200  # Number of 3D trajectory points to keep

# GPS-SLAM Fusion settings
ENABLE_GPS_FUSION = True  # Enable GPS-SLAM sensor fusion
GPS_WEIGHT = 0.3  # Weight for GPS vs SLAM (0.0 = pure SLAM, 1.0 = pure GPS)
DRIFT_CORRECTION_THRESHOLD = 5.0  # Maximum allowed drift before correction (meters)
GPS_UPDATE_FREQUENCY = 1.0  # GPS update frequency (seconds)
MIN_GPS_ACCURACY = 5.0  # Minimum GPS accuracy to use measurement (meters)
