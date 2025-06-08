# %% 
# Setup & Imports
import carla
import time
import threading
import pandas as pd
import sys
import random
import os
import joblib
from sklearn.preprocessing import StandardScaler

os.makedirs('output/rgb', exist_ok=True)

# True ➝ spoofing affects route
affect_steering = False  # SET TO True IN SECOND SCRIPT

# LOF Model Setup
lof_model = joblib.load('/content/drive/MyDrive/thesis/implementation/lof_model.pkl')
scaler = joblib.load('/content/drive/MyDrive/thesis/implementation/lof_scaler.pkl')
WINDOW_SIZE = 10
recent_features = []
spoof_detected = False

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
        return

    image_path = f'output/rgb/{image.frame:06d}.png'
    image.save_to_disk(image_path)
    sensor_data['rgb_frame'] = image.frame

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
    spoofing_active = True
    print("🚨 Spoofing activated.")
    if affect_steering:
        new_target = random.choice(map.get_spawn_points()).location
        agent.set_destination(new_target)
        print("🎯 Destination altered due to spoofing.")

threading.Thread(target=spoofing_trigger, args=(agent,)).start()

# %% 
# Main simulation + LOF detection
try:
    print("🚒 Running simulation with detection...")

    while True:
        world.tick()
        control = agent.run_step()
        vehicle.apply_control(control)

        velocity = vehicle.get_velocity()
        speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

        # Collect and process features
        if sensor_data['gnss'] and sensor_data['imu'] and sensor_data['rgb_frame'] is not None:
            diff_x = 0.0  # Placeholder for GPS-free position difference
            diff_y = 0.0

            feature = [
                diff_x, diff_y,
                sensor_data['imu']['accel_x'], sensor_data['imu']['accel_y'], sensor_data['imu']['accel_z'],
                sensor_data['imu']['gyro_x'], sensor_data['imu']['gyro_y'], sensor_data['imu']['gyro_z'],
                speed,
                control.steer,
                control.throttle,
                control.brake
            ]

            recent_features.append(feature)
            if len(recent_features) > WINDOW_SIZE:
                recent_features.pop(0)

            if len(recent_features) == WINDOW_SIZE and not spoof_detected:
                X_window = scaler.transform(recent_features)
                X_input = X_window.flatten().reshape(1, -1)
                prediction = lof_model.predict(X_input)[0]

                if prediction == -1:
                    print("🚨 Spoofing Detected! Stopping vehicle.")
                    spoof_detected = True

                    # 🚨 Display alert in CARLA UI
                    world.debug.draw_string(
                        vehicle.get_transform().location + carla.Location(z=3),
                        '🚨 Spoofing Detected!',
                        draw_shadow=True,
                        color=carla.Color(255, 0, 0),
                        life_time=2.0,
                        persistent_lines=False
                    )

                    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
                    time.sleep(2)
                    agent.set_destination(end_loc)
                    spoof_detected = False
                    print("📍 Resuming normal route.")

            # Record data
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

        # Update spectator view
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
