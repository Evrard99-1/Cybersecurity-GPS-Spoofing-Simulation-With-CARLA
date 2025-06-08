#%%
# # spoof_gps.py

import carla

spoofed_latitude = 52.000000
spoofed_longitude = 4.000000

def spoof_callback(event):
    print(f"[GPS] Real:    Lat={event.latitude:.6f}, Long={event.longitude:.6f}")
    print(f"[GPS] Spoofed: Lat={spoofed_latitude:.6f}, Long={spoofed_longitude:.6f}")

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

#%%
# Find GNSS sensor
gnss_sensor = None
for actor in world.get_actors().filter('sensor.other.gnss'):
    gnss_sensor = actor
    break

if not gnss_sensor:
    print("❌ No GNSS sensor found.")
else:
    gnss_sensor.listen(spoof_callback)
    print("📡 GPS spoofing active. Press Ctrl+C to stop.")

    try:
        while True:
            world.wait_for_tick()
    except KeyboardInterrupt:
        print("🛑 Spoofing stopped.")
        gnss_sensor.stop()
