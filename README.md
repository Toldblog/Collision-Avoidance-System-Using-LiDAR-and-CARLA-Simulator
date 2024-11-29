# Collision-Avoidance-System-Using-LiDAR-and-CARLA-Simulator
The system utilizes 3D point cloud data to detect vehicles on the road, analyze their movements across frames, and predict dangerous areas for collision avoidance

# Overview
This project involves developing a Collision Avoidance System for autonomous vehicles, leveraging the CARLA simulator and LiDAR sensor data. The system utilizes 3D point cloud data to detect vehicles on the road, analyze their movements across frames, and predict dangerous areas for collision avoidance. The final system identifies and marks hazardous regions to guide the self-driving car in avoiding potential collisions.

# Features
* LiDAR Data Processing: Captures 3D point cloud data from CARLA simulator's LiDAR sensors and filters out road points for analysis.
* Vehicle Detection: Detects and isolates vehicles present on the road by converting 3D points to 2D coordinates.
* Movement Analysis: Tracks and predicts vehicle movement through a novel method, as detailed in the attached methodology document.
* Collision Avoidance: Identifies and visualizes dangerous areas in real-time, ensuring the autonomous vehicle can navigate safely.
* Data Visualization: Provides 2D visualizations of vehicle movements and dangerous zones using Open3D and Matplotlib.

# Methodology

* LiDAR Data Acquisition: The LiDAR sensor captures 3D point clouds from the simulated environment in CARLA.
* Road Filtering: Points representing the road are identified and separated from the data.
* Vehicle Detection: Clusters of points above the road surface are detected using the DBSCAN clustering algorithm.

<div align=center>
  <img src="/images/3D_bbox.png" width="300" />
  <img src="/images/3D_visualize.png" width="500"/> 
</div>
  
* Movement Prediction: Tracks the movement of detected vehicles across consecutive frames by comparing clusters. Calculates the movement vectors for each vehicle based on the strategy described in Movement Prediction Methodology.

<div align=center>
  <img src="/images/Movement_Prediction_2.png" width="480"/> 
  <img src="/images/Movement_Prediction.png" width="300" />
</div>

* Collision Detection: Predicts dangerous areas by extrapolating vehicle paths and generating collision risk zones.

<div align=center>
  <img src="/images/Dangerous_areas.png" width="200"/> 
  <img src="/images/Binary_image.png" width="200" />
</div>

# Tools and Libraries

* CARLA Simulator: High-fidelity simulator for testing autonomous driving systems.
* Open3D: For 3D data visualization and point cloud processing.
* DBSCAN: Density-based clustering for detecting vehicles in point cloud data.
* Matplotlib: For 2D data visualization and road boundary mapping.
* NumPy: For numerical computations and data manipulation.
* OpenCV: For image processing and video generation.

# System Requirements
* Python 3.7.2+
* CARLA Simulator (version 0.9.15)
* Required Python libraries:
```
pip install numpy open3d scipy matplotlib opencv-python scikit-learn
```
