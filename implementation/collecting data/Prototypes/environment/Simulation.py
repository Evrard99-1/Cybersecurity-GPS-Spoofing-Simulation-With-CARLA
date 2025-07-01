# %%
# Cell 1: Imports and Setup
import glob
import os
import sys
import time
import argparse
import random
import carla

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# %%
# Cell 2: Argument parsing (set arguments manually if desired)
class Args:
    host = '127.0.0.1'
    port = 2000
    tm_port = 6000
    sync = True  # Change to False if synchronous mode is not desired

args = Args()

# %%
# Cell 3: Connect to CARLA and set spectator
client = carla.Client(args.host, args.port)
client.set_timeout(10.0)
world = client.get_world()

spectator = world.get_spectator()
spectator.set_transform(
    carla.Transform(carla.Location(x=0, y=0, z=50), carla.Rotation(pitch=-90))
)

# %%
# Cell 4: Traffic Manager setup
traffic_manager = client.get_trafficmanager(args.tm_port)
traffic_manager.set_global_distance_to_leading_vehicle(2.5)
print(f"Traffic Manager bound to port {args.tm_port}")
time.sleep(2)

# %%
# Cell 5: Enable synchronous mode (optional)
settings = world.get_settings()
if args.sync:
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

# %%
# Cell 6: Retrieve vehicle blueprint and GPS spoofing coordinates
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')[0]

spoofed_latitude = 52.0
spoofed_longitude = 4.0

# %%
# Cell 7: Spawn vehicles and attach GPS sensors
num_vehicles = 2
vehicles = []
gps_sensors = []
spawn_location = carla.Location(x=0, y=0, z=1)

def gps_callback(event, vehicle_id):
    original_lat = event.latitude
    original_long = event.longitude
    print(f"[Vehicle {vehicle_id}] Original GPS: Latitude={original_lat}, Longitude={original_long}")
    print(f"[Vehicle {vehicle_id}] Spoofed GPS: Latitude={spoofed_latitude}, Longitude={spoofed_longitude}")

for i in range(num_vehicles):
    spawn_point = carla.Transform(
        location=carla.Location(
            x=spawn_location.x + i * 10,
            y=spawn_location.y,
            z=spawn_location.z
        ),
        rotation=carla.Rotation(yaw=0)
    )

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle:
        vehicle.set_autopilot(True, traffic_manager.get_port())
        vehicles.append(vehicle)
        print(f"Vehicle {i} successfully spawned at {spawn_point.location}")

        gps_bp = blueprint_library.find('sensor.other.gnss')
        gps_transform = carla.Transform(carla.Location(x=1.0, z=2.8))
        gps_sensor = world.spawn_actor(gps_bp, gps_transform, attach_to=vehicle)

        gps_sensor.listen(lambda event, id=vehicle.id: gps_callback(event, id))
        gps_sensors.append(gps_sensor)
    else:
        print("Spawn failed due to collision or other issues.")

# %%
# Cell 8: Run simulation loop
try:
    while True:
        if args.sync:
            world.tick()
        else:
            world.wait_for_tick()
except KeyboardInterrupt:
    print("Simulation interrupted by user")
finally:
    print("Destroying actors...")
    for gps_sensor in gps_sensors:
        gps_sensor.stop()
        gps_sensor.destroy()
    for vehicle in vehicles:
        vehicle.destroy()
    if args.sync:
        settings.synchronous_mode = False
        world.apply_settings(settings)
    print("Simulation ended")
