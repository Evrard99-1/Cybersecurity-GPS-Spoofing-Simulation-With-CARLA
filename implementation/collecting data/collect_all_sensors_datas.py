# %% 
# Setup & Imports
import carla
import time
import threading
import pandas as pd
import sys
import random
import os
os.makedirs('output/rgb', exist_ok=True)


# True ➝ spoofing affects route
affect_steering = False  # SET TO True IN SECOND SCRIPT

# %% 
# Connect & Setup
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
map = world.get_map()
blueprints = world.get_blueprint_library()
spawn_points = map.get_spawn_points()

# %% 
# Destroy existing actors
for a in world.get_actors().filter('*vehicle*'): a.destroy()
for s in world.get_actors().filter('*sensor*'): s.destroy()

# %% 
# Global Variables
data_records = []
spoofing_active = False
spoofed_latitude = 52.000000
spoofed_longitude = 4.000000
frame_id = 0

# %% 
# Firetruck
vehicle_bp = blueprints.filter('*firetruck*')[0]
vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
if not vehicle:
    print("❌ Firetruck failed to spawn.")
    exit()
print("✅ Firetruck spawned.")

# %% 
# Spectator
spectator = world.get_spectator()
spectator.set_transform(carla.Transform(
    vehicle.get_transform().transform(carla.Location(x=-8, z=3)),
    vehicle.get_transform().rotation))

# %% 
# Sensor setup
sensor_list = []
sensor_data = {
    'gnss': None,
    'imu': None,
    'rgb_frame': None
}

# GNSS
def gps_callback(data):
    sensor_data['gnss'] = {
        'latitude': spoofed_latitude if spoofing_active else data.latitude,
        'longitude': spoofed_longitude if spoofing_active else data.longitude
    }

gnss = world.spawn_actor(blueprints.find('sensor.other.gnss'), carla.Transform(), attach_to=vehicle)
gnss.listen(gps_callback)
sensor_list.append(gnss)

# IMU
def imu_callback(data):
    sensor_data['imu'] = {
        'accel_x': data.accelerometer.x,
        'accel_y': data.accelerometer.y,
        'accel_z': data.accelerometer.z,
        'gyro_x': data.gyroscope.x,
        'gyro_y': data.gyroscope.y,
        'gyro_z': data.gyroscope.z
    }

imu = world.spawn_actor(blueprints.find('sensor.other.imu'), carla.Transform(), attach_to=vehicle)
imu.listen(imu_callback)
sensor_list.append(imu)

# RGB Camera
def rgb_callback(image):
    if not (sensor_data['gnss'] and sensor_data['imu']):
        return  # Skip if GNSS or IMU not ready

    image_path = f'output/rgb/{image.frame:06d}.png'
    image.save_to_disk(image_path)
    
    velocity = vehicle.get_velocity()
    speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
    global frame_id

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

camera_bp = blueprints.find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=2, z=3), carla.Rotation(pitch=-10))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
camera.listen(rgb_callback)
sensor_list.append(camera)


# %% 
# Route planning
sys.path.append('D:/carla/PythonAPI/carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.basic_agent import BasicAgent

grp = GlobalRoutePlanner(map, 2.0)
start_loc = carla.Location(x=50.477512, y=141.135620, z=0.001844)
end_loc = carla.Location(x=-64.644844, y=24.471010, z=0.600000)
route = grp.trace_route(start_loc, end_loc)

for wp, _ in route:
    world.debug.draw_arrow(wp.transform.location,
                           wp.transform.location + carla.Location(z=0.5),
                           0.1, 0.3,
                           carla.Color(0, 255, 0),
                           60.0, True)

agent = BasicAgent(vehicle)
agent.set_destination(end_loc)

# %% 
# Spoofing timer
def spoofing_trigger(agent, delay=10):
    global spoofing_active
    print(f"🕐 Spoofing starts in {delay}s...")
    time.sleep(delay)

    # Spoofing attack
    spoofing_active = True   # SET TO True to activate spoofing attack

    print("🚨 Spoofing activated.")
    if affect_steering:
        new_target = random.choice(map.get_spawn_points()).location
        agent.set_destination(new_target)
        print("🎯 Destination altered due to spoofing.")

threading.Thread(target=spoofing_trigger, args=(agent,)).start()

# %% 
# Main simulation + collection loop
try:
    print("🚒 Running simulation...")
    while True:
        world.tick()
        control = agent.run_step()
        vehicle.apply_control(control)

        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

        if sensor_data['gnss'] and sensor_data['imu'] and sensor_data['rgb_frame'] is not None:
            record = {
                'frame_id': frame_id,
                'image_num': sensor_data['rgb_frame'],
                'latitude': sensor_data['gnss']['latitude'],
                'longitude': sensor_data['gnss']['longitude'],
                'accel_x': sensor_data['imu']['accel_x'],
                'accel_y': sensor_data['imu']['accel_y'],
                'accel_z': sensor_data['imu']['accel_z'],
                'gyro_x': sensor_data['imu']['gyro_x'],
                'gyro_y': sensor_data['imu']['gyro_y'],
                'gyro_z': sensor_data['imu']['gyro_z'],
                'steering_angle': control.steer,
                'throttle': control.throttle,
                'brake': control.brake,
                'speed': speed,
                'label': 'spoofed' if spoofing_active else 'normal'
            }
            data_records.append(record)
            frame_id += 1

        spectator.set_transform(carla.Transform(
            vehicle.get_transform().transform(carla.Location(x=-8, z=3)),
            vehicle.get_transform().rotation))

        if agent.done():
            print("✅ Destination reached.")
            break
except KeyboardInterrupt:
    print("🛑 Interrupted.")

# %% 
# Save and cleanup
df = pd.DataFrame(data_records)
df.to_csv('combined_data.csv', index=False)
print("💾 Data saved to combined_data.csv")

for s in sensor_list:
    s.stop()
    s.destroy()
vehicle.destroy()
print("✅ Cleanup complete.")
