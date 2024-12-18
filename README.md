# Collision-Avoidance-System-Using-LiDAR-and-CARLA-Simulator
The system utilizes 3D point cloud data to detect vehicles on the road, analyze their movements across frames, and predict dangerous areas for collision avoidance

# Overview
This project involves developing a Collision Avoidance System for autonomous vehicles, leveraging the CARLA simulator and LiDAR sensor data. The system utilizes 3D point cloud data to detect vehicles on the road, analyze their movements across frames, and predict dangerous areas for collision avoidance. The final system identifies and marks hazardous regions to guide the self-driving car in avoiding potential collisions.

# Workflow
<div align=center>
  <img src="/images/workflow.png" width="800" />
</div>

* From the LiDAR sensor, we collect 3D cloud point data that surrounds our car.
  
<div align=center>
  <img src="/images/3d.png" width="400" />
</div>

* Filtering out road point.

<div align=center>
  <img src="/images/road_3d.png" width="400" />
</div>

* Filtering out a set of clusters of points that are located on the surface of the road (vehicles).
  
<div align=center>
  <img src="/images/vehicle_3d.png" width="400" />
</div>

* Combine the road and vehicle sets of points together and convert them into a 2D data type.
  
<div align=center>
  <img src="/images/combine.png" width="400" />
  <img src="/images/2d.png" width="240" />
</div>

* Applying the Movement Prediction algorithm to find out the movement of a cluster of points.

<div align=center>
  <img src="/images/Movement_Prediction_2.png" width="600"/> 
  <img src="/images/movement_prediction_2d.png" width="375"/> 
</div>

* When we already get the movement direction, expand the danger zone (after a few seconds these vehicles will go to this area).
  
<div align=center>
  <img src="/images/danger_zone_2d.png" width="300" />
</div>


* Find the safe quadrant that our car intends to move to by finding out which part (top, bottom, left, or right) has the largest number of danger points in our car bounding box.


<div align=center>
  <img src="/images/safe_quadrant.png" width="300" />
</div>

* After that, create a sliding window that slides in the safe quadrant to find the free space (no obstacle in it, and our car can move to that space safely).

<div align=center>
  <img src="/images/3_red.png" width="300" />
  <img src="/images/2_red.png" width="300"/> 
  <img src="/images/1_green.png" width="300"/> 
</div>

* The last step is to control our car to the center point of that free space.

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
