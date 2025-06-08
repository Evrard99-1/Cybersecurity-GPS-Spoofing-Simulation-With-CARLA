# %% 
# Setup & Imports
import carla
import time
import threading
import pandas as pd
import sys
import random
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

os.makedirs('output/rgb', exist_ok=True)

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
sequence_buffer = []

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
original_dest = end_loc

# %% 
# Spoofing trigger
def spoofing_trigger(agent, delay=10):
    global spoofing_active
    print(f"🕐 Spoofing starts in {delay}s...")
    time.sleep(delay)
    spoofing_active = True
    print("🚨 Spoofing activated.")

threading.Thread(target=spoofing_trigger, args=(agent,)).start()

# %% 
# Load Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(32*16*16 + 5, 128, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, img_seq, sensor_seq):
        batch_size, seq_len, c, h, w = img_seq.shape
        img_seq = img_seq.view(-1, c, h, w)
        img_feat = self.cnn(img_seq)
        img_feat = img_feat.view(batch_size, seq_len, -1)
        combined = torch.cat([img_feat, sensor_seq], dim=2)
        lstm_out, _ = self.lstm(combined)
        return self.fc(lstm_out[:, -1, :])

model = CNNLSTM()
model.load_state_dict(torch.load('/content/drive/MyDrive/thesis/implementation/pca_cnn_lstm_model.pth', map_location=DEVICE))
model.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

scaler = StandardScaler()
pca = PCA(n_components=5)

# %% 
# Main loop with UI alerts
try:
    print("🚒 Simulation running...")
    spoof_detected = False
    spoof_pause_done = False

    while True:
        world.tick()
        control = agent.run_step()

        if sensor_data['gnss'] and sensor_data['imu'] and sensor_data['rgb_frame'] is not None:
            img_path = f'output/rgb/{sensor_data["rgb_frame"]:06d}.png'
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)

            velocity = vehicle.get_velocity()
            speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

            sensor_vector = np.array([
                sensor_data['imu']['accel_x'],
                sensor_data['imu']['accel_y'],
                sensor_data['imu']['accel_z'],
                sensor_data['imu']['gyro_x'],
                sensor_data['imu']['gyro_y'],
                sensor_data['imu']['gyro_z'],
                speed,
                control.steer,
                control.throttle,
                control.brake
            ]).reshape(1, -1)

            sensor_vector = scaler.fit_transform(sensor_vector)
            sensor_pca = pca.fit_transform(sensor_vector)

            sequence_buffer.append((img_tensor, torch.tensor(sensor_pca[0], dtype=torch.float32)))
            if len(sequence_buffer) > 5:
                sequence_buffer.pop(0)

            if len(sequence_buffer) == 5 and not spoof_detected:
                imgs = torch.stack([x[0] for x in sequence_buffer]).unsqueeze(0).to(DEVICE)
                sensors = torch.stack([x[1] for x in sequence_buffer]).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    pred = model(imgs, sensors).argmax(dim=1).item()
                    if pred == 1:
                        print("🚨 Spoofing detected!")
                        spoof_detected = True
                        spoofing_active = True
                        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

                        world.debug.draw_string(vehicle.get_location(), "[SPOOFING DETECTED]", life_time=2.0, color=carla.Color(255, 0, 0))
                        time.sleep(2)

                        spoofing_active = False
                        agent.set_destination(original_dest)
                        print("✅ Route reset.")
                        spoof_pause_done = True

            if not spoof_detected or spoof_pause_done:
                vehicle.apply_control(control)

            # Save data
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
    print("🛑 Simulation interrupted.")

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
