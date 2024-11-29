import numpy as np
from matplotlib import cm

# CARLA SIMULATOR
PORT = 2000
TIMEOUT_MAX = 5.0  # seconds
# simulation frame rate of 1 / delta = 1 / 0.05 = 20 frames per second (FPS)
DELTA = 0.05

CAR_MODEL = "*model3*"

# DBSCAN parameters
EPS = 0.5
MIN_SAMPLES = 10
# Exclude any cluster have points have z higher than this value
CLUSTER_UPPER_EXCLUDE = 2

# Default scale factor for arrows
ARROW_SCALE_FACTOR = 15.0

# Default frame rate for video
FRAME_RATE = 20

# Velocity threshold for drawing arrows (in km/h)
VELOCITY_THRESHOLD = 4

# Apply EMA in update the movement diretion when drawing arrow
UPDATED_DIRECTION_PARAM = 0.7
PREVIOUS_DIRECTION_PARAM = 0.3

# Distance threshold for tracking centroid of cluster (in meters)
DIST_THRESHOLD_TRACKING = 1

# LiDAR sensor configuration
LIDAR_UPPER_FOV = 10.0
LIDAR_LOWER_FOV = -30.0
LIDAR_CHANNELS = 64
LIDAR_RANGE = 100.0
LIDAR_ROTATION_FREQUENCY = 20
LIDAR_POINTS_PER_SECOND = 1000000

LIDAR_POSITION_HEIGHT = 2  # z = 2

# Vehicle speed control parameters
PREFERRED_SPEED = 30  # Desired speed in km/h
SPEED_THRESHOLD = 2

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

# Accept cloud point have height lower than 4 meter
HEIGHT_UPPER_FILTER = 4 - LIDAR_POSITION_HEIGHT
# Accept cloud point have height higher than 0.3 meter
HEIGHT_LOWER_FILTER = 0.3 - LIDAR_POSITION_HEIGHT

ZOOM_LIDAR_BIRD_EYE_PARAM = 1

SECONDS_PER_EPISODE = 15

MAX_STEER_DEGREES = 40
