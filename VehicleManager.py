from helpers.utils import *
from constants import *
import sys
import weakref

import carla

import sys


class VehicleManager:
    def __init__(self, world=None, spawn_transform=None, attach_lidar=False, attach_collision=False, autopilot=True, vehicle_bp_filter=CAR_MODEL):
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_transform = spawn_transform
        self.vehicle_bp_filter = vehicle_bp_filter
        self.vehicle_bp = self.blueprint_library.filter(vehicle_bp_filter)[0]
        self.vehicle = self.world.spawn_actor(
            self.vehicle_bp, self.spawn_transform)
        self.vehicle.set_autopilot(autopilot)

        self.attach_lidar = attach_lidar
        self.attach_collision = attach_collision

        self.point_cloud_data = []

        self.original_transform = spawn_transform
        self.collision_sensor = None
        self.collision_detected = False
        self.collision_location = None

        if attach_collision:
            self.attach_collision_sensor()

        if attach_lidar:
            self.attach_lidar_sensor()

    def attach_lidar_sensor(self):
        self.lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('upper_fov', str(LIDAR_UPPER_FOV))
        self.lidar_bp.set_attribute('lower_fov', str(LIDAR_LOWER_FOV))
        self.lidar_bp.set_attribute('channels', str(LIDAR_CHANNELS))
        self.lidar_bp.set_attribute('range', str(LIDAR_RANGE))
        self.lidar_bp.set_attribute(
            'rotation_frequency', str(LIDAR_ROTATION_FREQUENCY))
        self.lidar_bp.set_attribute(
            'points_per_second', str(LIDAR_POINTS_PER_SECOND))
        self.lidar_transform = carla.Transform(
            carla.Location(x=0, z=LIDAR_POSITION_HEIGHT))
        self.lidar = self.world.spawn_actor(
            self.lidar_bp, self.lidar_transform, attach_to=self.vehicle)
        self.lidar.listen(lambda data: self.lidar_callback(data))

    def lidar_callback(self, point_cloud_data):
        data = np.copy(np.frombuffer(
            point_cloud_data.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        # Isolate the intensity and compute a color for it
        intensity = data[:, -1]
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

        points = data[:, :-1]
        points[:, :1] = -points[:, :1]
        # Update the point cloud data
        point_cloud_data.points = o3d.utility.Vector3dVector(points)
        point_cloud_data.colors = o3d.utility.Vector3dVector(int_color)
        self.point_cloud_data.append(points)

    def attach_collision_sensor(self):
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle)
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(
            lambda event: VehicleManager._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.collision_detected = True
        self.collision_location = event.transform.location

    def apply_control(self, throttle, steer):
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=throttle, steer=steer))

    def respawn(self):
        self.destroy()

        self.vehicle = self.world.try_spawn_actor(
            self.vehicle_bp, self.original_transform)
        if self.vehicle is None:
            print("Failed to respawn vehicle.")
            return False
        self.collision_detected = False
        self.collision_location = None

        if self.attach_collision:
            self.attach_collision_sensor()

        if self.attach_lidar == True:
            self.attach_lidar_sensor()

    def destroy(self):
        if self.lidar:
            self.lidar.destroy()
            self.lidar = None
        if self.collision_sensor:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision_sensor = None
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None
