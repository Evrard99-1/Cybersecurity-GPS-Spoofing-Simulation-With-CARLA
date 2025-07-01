# full_spoof_detection_with_mlp.py
# Real-time GPS spoof detection in Carla using saved SKLearn MLP GPS-IDS model

# === SETUP & IMPORTS ===
import carla
import time, threading, random, os, sys
import numpy as np
import cv2
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# make sure Carla's PythonAPI is on the path (adjust as needed)
sys.path.append('D:/carla/PythonAPI')
sys.path.append('D:/carla/PythonAPI/carla')

# ==== CONFIG ====  
DEVICE         = 'cpu'  # SKLearn runs on CPU
gps_model_path = 'gps_ids_mlp_model.pkl'
threshold_path = 'gps_ids_threshold.txt'
VIDEO_DIR      = 'video_NN'
FPS            = 20

# ==== LOAD GPS-IDS MODEL & THRESHOLD ====  
if not os.path.isfile(gps_model_path):
    raise FileNotFoundError(f"Model file not found: {gps_model_path}")
if not os.path.isfile(threshold_path):
    raise FileNotFoundError(f"Threshold file not found: {threshold_path}")

gps_model = joblib.load(gps_model_path)
with open(threshold_path, 'r') as f:
    THRESHOLD = float(f.read().strip())
print(f"Loaded MLP GPS-IDS model and threshold={THRESHOLD:.3f}")

# ==== CARLA SETUP ====
client      = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world       = client.get_world()
blueprints  = world.get_blueprint_library()
spawn_pts   = world.get_map().get_spawn_points()

# cleanup leftover actors
for a in world.get_actors().filter('*vehicle*'): a.destroy()
for s in world.get_actors().filter('*sensor*'): s.destroy()

# make output dirs
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(f'{VIDEO_DIR}/frames', exist_ok=True)

# === LABEL SPAWN POINTS ===
print("\U0001F4CD Drawing enlarged label numbers at each spawn point...")
for idx, sp in enumerate(spawn_points):
    loc = sp.location + carla.Location(z=2.5)
    offsets = [(-0.15, 0), (0.15, 0), (0, -0.15), (0, 0.15), (0, 0)]
    for dx, dy in offsets:
        offset_loc = loc + carla.Location(x=dx, y=dy)
        world.debug.draw_string(offset_loc, f"#{idx}", draw_shadow=True,
                                color=carla.Color(r=255, g=255, b=0), life_time=100.0, persistent_lines=True)
print(f"✅ {len(spawn_points)} bold label numbers displayed in simulator.")


# === USER INPUT FOR SPAWN ===
start_i = int(input("Enter START spawn index: "))
spoof_i = int(input("Enter SPOOFED GPS spawn index: "))
start_tf  = spawn_pts[start_i]
spoof_loc = spawn_pts[spoof_i].location

# === SPAWN VEHICLE & AUTOPILOT ===
veh_bp  = blueprints.find('vehicle.lincoln.mkz_2020')
vehicle = world.spawn_actor(veh_bp, start_tf)
vehicle.set_autopilot(True)

# spectator follow thread
running = True
def follow_vehicle():
    spectator = world.get_spectator()
    while running and vehicle.is_alive:
        t = vehicle.get_transform()
        spectator.set_transform(
            carla.Transform(t.transform(carla.Location(x=-6, z=3)), t.rotation)
        )
        time.sleep(0.05)
threading.Thread(target=follow_vehicle, daemon=True).start()

# === GLOBAL STATE & DATA ===
spoofing_active   = False
spoofed_lat, spoofed_lon = spoof_loc.x, spoof_loc.y
frame_id          = 0
video_writers     = {}
data_records      = []
sensor_data       = {'gnss': None, 'imu': None}
sensor_list       = []

# === MANUAL STOP THREAD ===
def stop_loop():
    global running
    while running:
        cmd = input("Type 'q' + ENTER to quit → ").strip().lower()
        if cmd == 'q':
            running = False
            print("🛑 Stopping simulation…")
            break
threading.Thread(target=stop_loop, daemon=True).start()

# === RANDOM SPOOF THREAD ===
def random_spoof_loop():
    global spoofing_active
    while running:
        time.sleep(random.uniform(5,15))
        spoofing_active = True
        print("🚨 Random Spoof ON for 5s")
        time.sleep(5)
        spoofing_active = False
        print("✅ Random Spoof OFF")
threading.Thread(target=random_spoof_loop, daemon=True).start()

# === SENSOR CALLBACKS ===
def gps_callback(data):
    sensor_data['gnss'] = {
        'latitude':  spoofed_lat   if spoofing_active else data.latitude,
        'longitude': spoofed_lon   if spoofing_active else data.longitude
    }

def imu_callback(data):
    sensor_data['imu'] = {
        'accel_x': data.accelerometer.x,
        'accel_y': data.accelerometer.y,
        'accel_z': data.accelerometer.z,
        'gyro_x':  data.gyroscope.x,
        'gyro_y':  data.gyroscope.y,
        'gyro_z':  data.gyroscope.z
    }

# === MAIN SENSOR CALLBACK ===
def rgb_callback(image):
    global frame_id
    if not (sensor_data['gnss'] and sensor_data['imu']):
        return

    # convert BGRA→BGR
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    rgb_img = arr[:, :, :3][:, :, ::-1].copy()

    # compute speed (km/h)
    v = vehicle.get_velocity()
    speed = 3.6 * (v.x**2 + v.y**2 + v.z**2)**0.5

    # assemble features
    ctrl = vehicle.get_control()
    feat = np.array([
        sensor_data['imu']['accel_x'], sensor_data['imu']['accel_y'], sensor_data['imu']['accel_z'],
        sensor_data['imu']['gyro_x'], sensor_data['imu']['gyro_y'], sensor_data['imu']['gyro_z'],
        speed, ctrl.steer, ctrl.throttle, ctrl.brake
    ], dtype=np.float32)

    # predict
    prob = gps_model.predict_proba(feat.reshape(1,-1))[0,1]
    label = 'spoofed' if prob > THRESHOLD else 'normal'
    conf = prob if label=='spoofed' else (1-prob)

    print(f"[Frame {frame_id}] {label.upper()} ({conf:.2f})")

    # overlay label
    color = (255,0,0) if label=='spoofed' else (0,255,0)
    cv2.putText(rgb_img, f"{label.upper()} {conf*100:.1f}%", (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # save frame/video
    if 'vid' not in video_writers:
        h,w = image.height, image.width
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writers['vid'] = cv2.VideoWriter(f"{VIDEO_DIR}/rgb.avi", fourcc, FPS, (w,h))
    video_writers['vid'].write(rgb_img)
    cv2.imwrite(f"{VIDEO_DIR}/frames/{frame_id:05d}.jpg", rgb_img)

    # log
    rec = {'frame_id': frame_id,
           'latitude': sensor_data['gnss']['latitude'], 'longitude': sensor_data['gnss']['longitude'],
           'label': 'spoofed' if spoofing_active else 'normal',
           'predicted_label': label, 'spoof_probability': prob}
    keys = ['accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z','speed','steering_angle','throttle','brake']
    for k,v in zip(keys, feat): rec[k]=v
    data_records.append(rec)
    frame_id += 1

# === ATTACH SENSORS ===
for bp, callback in [
    (blueprints.find('sensor.other.gnss'), gps_callback),
    (blueprints.find('sensor.other.imu'), imu_callback),
    (blueprints.find('sensor.camera.rgb'), rgb_callback)
]:
    tf = carla.Transform() if bp.id.endswith('gnss') or bp.id.endswith('imu') else carla.Transform(carla.Location(x=1.5,z=2.4))
    actor = world.spawn_actor(bp, tf, attach_to=vehicle)
    actor.listen(callback)
    sensor_list.append(actor)

# === MAIN LOOP ===
try:
    while running and vehicle.is_alive:
        world.tick()
        time.sleep(0.05)
except KeyboardInterrupt:
    running = False
    print("Interrupted")

# === CLEANUP & SAVE CSV & PLOT ===
all_keys = set().union(*(r.keys() for r in data_records))
cleaned  = [{k: rec.get(k, np.nan) for k in all_keys} for rec in data_records]
df_out = pd.DataFrame(cleaned, columns=sorted(all_keys))
df_out.to_csv('combined_data_new_test.csv', index=False)
print("💾 combined_data_new_test.csv saved.")

# compute detection metrics & plot
df_out['ground_truth'] = (df_out['label']=='spoofed').astype(int)
