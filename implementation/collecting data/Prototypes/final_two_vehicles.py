# === SETUP & IMPORTS ===
import carla
import time
import threading
import pandas as pd
import random
import os
import sys
import numpy as np
import cv2

# Ensure output directories exist
os.makedirs('output/video', exist_ok=True)

sys.path.append('D:/carla/PythonAPI')
sys.path.append('D:/carla/PythonAPI/carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.behavior_agent import BehaviorAgent

# === CONNECTION ===
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
map = world.get_map()
blueprints = world.get_blueprint_library()
spawn_points = map.get_spawn_points()

# === CLEANUP OLD ACTORS ===
for a in world.get_actors().filter('*vehicle*'): a.destroy()
for s in world.get_actors().filter('*sensor*'): s.destroy()

# === LABEL SPAWN POINTS ===
print("\U0001F4CD Drawing enlarged label numbers at each spawn point...")
for idx, sp in enumerate(spawn_points):
    loc = sp.location + carla.Location(z=2.5)
    offsets = [(-0.15, 0), (0.15, 0), (0, -0.15), (0, 0.15), (0, 0)]
    for dx, dy in offsets:
        offset_loc = loc + carla.Location(x=dx, y=dy)
        world.debug.draw_string(offset_loc, f"#{idx}", draw_shadow=True,
                                color=carla.Color(r=255, g=255, b=0), life_time=600.0, persistent_lines=True)
print(f"✅ {len(spawn_points)} bold label numbers displayed in simulator.")

# === USER INPUTS ===
start_index = int(input("\nEnter index for START location: "))
spoof_index = int(input("Enter index for SPOOFED GPS location: "))
end_index = int(input("Enter index for FINAL DESTINATION: "))

start_transform = spawn_points[start_index]
spoof_location = spawn_points[spoof_index].location
end_transform = spawn_points[end_index]

print("\n=== Navigation Plan Summary ===")
print(f"Start Location:      {start_transform.location}")
print(f"Spoofed GPS Location:{spoof_location}")
print(f"Destination:         {end_transform.location}")

# === VEHICLE ===
vehicle_bp = blueprints.filter('vehicle.lincoln.mkz_2020')[0]
vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
vehicle.set_transform(carla.Transform(start_transform.location + carla.Location(x=1, z=10),
                                      carla.Rotation(yaw=start_transform.rotation.yaw)))
print("\U0001F697 Vehicle spawned.")

spectator = world.get_spectator()
spectator.set_transform(carla.Transform(
    vehicle.get_transform().transform(carla.Location(x=-5, z=2)),
    vehicle.get_transform().rotation))

# === SPOOFING SETUP ===
spoofing_active = False
spoofed_latitude = spoof_location.x
spoofed_longitude = spoof_location.y
frame_id = 0
video_fps = 20
video_writers = {}
data_records = []
sensor_data = { 'gnss': None, 'imu': None }
sensor_list = []
affect_steering = True

# === SENSOR CALLBACKS ===
def gps_callback(data):
    sensor_data['gnss'] = {
        'latitude': spoofed_latitude if spoofing_active else data.latitude,
        'longitude': spoofed_longitude if spoofing_active else data.longitude
    }

def imu_callback(data):
    sensor_data['imu'] = {
        'accel_x': data.accelerometer.x,
        'accel_y': data.accelerometer.y,
        'accel_z': data.accelerometer.z,
        'gyro_x': data.gyroscope.x,
        'gyro_y': data.gyroscope.y,
        'gyro_z': data.gyroscope.z
    }

def rgb_callback(image):
    global frame_id
    if not (sensor_data['gnss'] and sensor_data['imu']): return

    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    rgb_img = array[:, :, :3][:, :, ::-1]

    if 'rgb' not in video_writers:
        path = 'output/video/rgb_video.avi'
        video_writers['rgb'] = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), video_fps, (image.width, image.height))
    video_writers['rgb'].write(rgb_img)

    velocity = vehicle.get_velocity()
    speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5

    record = {
        'frame_id': frame_id,
        'image_num': image.frame,
        'latitude': sensor_data['gnss']['latitude'],
        'longitude': sensor_data['gnss']['longitude'],
        'accel_x': sensor_data['imu']['accel_x'],
        'accel_y': sensor_data['imu']['accel_y'],
        'accel_z': sensor_data['imu']['accel_z'],
        'gyro_x': sensor_data['imu']['gyro_x'],
        'gyro_y': sensor_data['imu']['gyro_y'],
        'gyro_z': sensor_data['imu']['gyro_z'],
        'steering_angle': vehicle.get_control().steer,
        'throttle': vehicle.get_control().throttle,
        'brake': vehicle.get_control().brake,
        'speed': speed,
        'label': 'spoofed' if spoofing_active else 'normal'
    }
    data_records.append(record)
    frame_id += 1

def depth_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    depth_img = cv2.applyColorMap(array[:, :, 0], cv2.COLORMAP_JET)

    if 'depth' not in video_writers:
        path = 'output/video/depth_video.avi'
        video_writers['depth'] = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), video_fps, (image.width, image.height))
    video_writers['depth'].write(depth_img)

def seg_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    seg_img = array[:, :, :3]

    if 'seg' not in video_writers:
        path = 'output/video/segmentation_video.avi'
        video_writers['seg'] = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), video_fps, (image.width, image.height))
    video_writers['seg'].write(seg_img)

def lidar_callback(point_cloud):
    points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
    lidar_img = np.zeros((600, 800, 3), dtype=np.uint8)
    for point in points:
        x = int(point[0] * 2 + 400)
        y = int(-point[1] * 2 + 300)
        if 0 <= x < 800 and 0 <= y < 600:
            lidar_img[y, x] = (0, 255, 0)

    if 'lidar' not in video_writers:
        path = 'output/video/lidar_video.avi'
        video_writers['lidar'] = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), video_fps, (800, 600))
    video_writers['lidar'].write(lidar_img)

# === SENSOR ATTACHMENT ===
gnss = world.spawn_actor(blueprints.find('sensor.other.gnss'), carla.Transform(), attach_to=vehicle)
imu = world.spawn_actor(blueprints.find('sensor.other.imu'), carla.Transform(), attach_to=vehicle)
camera_bp = blueprints.find('sensor.camera.rgb')
depth_bp = blueprints.find('sensor.camera.depth')
seg_bp = blueprints.find('sensor.camera.semantic_segmentation')
lidar_bp = blueprints.find('sensor.lidar.ray_cast')
camera_transform = carla.Transform(carla.Location(x=4, z=2), carla.Rotation(pitch=-10))

camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
depth = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
seg = world.spawn_actor(seg_bp, camera_transform, attach_to=vehicle)
lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(z=2)), attach_to=vehicle)

# === SENSOR LISTEN ===
gnss.listen(gps_callback)
imu.listen(imu_callback)
camera.listen(rgb_callback)
depth.listen(depth_callback)
seg.listen(seg_callback)
lidar.listen(lidar_callback)

sensor_list += [gnss, imu, camera, depth, seg, lidar]

# === ROUTE PLANNING AND DRIVING ===
agent = BehaviorAgent(vehicle, behavior='normal')
agent.set_destination(end_transform.location)

while vehicle.get_location().distance(end_transform.location) > 2.0:
    world.tick()
    control = agent.run_step()
    vehicle.apply_control(control)

# === CLEANUP & SAVE ===
pd.DataFrame(data_records).to_csv('combined_data.csv', index=False)
for s in sensor_list: s.stop(); s.destroy()
vehicle.destroy()
for writer in video_writers.values(): writer.release()
print("\u2705 Data and videos saved. All actors cleaned up.")
