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

# Timing settings
ASYNC_SLEEP_INTERVAL = 0.01
PYGAME_WAIT_TIME = 10
GUI_TIMEOUT = 0.1
TAKEOFF_WAIT_TIME = 5
CONTROL_LOOP_INTERVAL = 0.1

# Improved VO settings
MOTION_DETECTION_THRESHOLD = 5.0  # Skip VO if frame difference < this
MIN_FEATURES_FOR_VO = 50  # Minimum features needed for pose estimation
TEMPORAL_SMOOTHING_ALPHA = 0.7  # Pose smoothing factor (0.7 = 70% new, 30% old)
MAX_POSE_HISTORY = 5  # Number of poses to keep for smoothing

# SIFT feature detector settings
SIFT_FEATURES = 1000  # Maximum features to detect
SIFT_CONTRAST_THRESHOLD = 0.04  # Feature quality threshold
SIFT_EDGE_THRESHOLD = 10  # Edge response threshold

# Feature matching settings
RATIO_TEST_THRESHOLD = 0.7  # Lowe's ratio test threshold
RANSAC_THRESHOLD = 1.0  # RANSAC inlier threshold (pixels)
RANSAC_CONFIDENCE = 0.999  # RANSAC confidence level
