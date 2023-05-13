import cv2
import time
import VL53L1X
import os
import pandas as pd

# Function to capture image from all connected cameras
def capture_images(cameras):
    images = []
    for cam in cameras:
        ret, frame = cam.read()
        if not ret:
            print(f"Failed to grab frame from camera: {cam}")
            continue
        images.append(frame)
    return images

# Get list of connected cameras
def get_connected_cameras():
    cameras = []
    for i in range(10): # try for 10 cameras, adjust according to your needs
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            cameras.append(cap)
            cap.release()
        else:
            break
    return cameras

# Initialize LIDAR
tof = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)

# Initial setup for cameras
cameras = get_connected_cameras()

# Empty list to store data
data = []

# Capture images and LIDAR measurements and save to a directory
while True: # continuous loop
    tof.open()
    tof.start_ranging(1)  # Start ranging, 1 = Short Range, 2 = Medium Range, 3 = Long Range

    # Capture images
    images = capture_images(cameras)

    # Get LIDAR measurement
    lidar_measurement = tof.get_distance()

    tof.stop_ranging()
    tof.close()

    # Save images and append measurement to data
    img_paths = []
    for idx, img in enumerate(images):
        img_path = f'images/{time.time()}_{idx}.jpg'
        cv2.imwrite(img_path, img)
        img_paths.append(img_path)

    data.append({
        'lidar_measurement': lidar_measurement,
        'img_paths': img_paths
    })

    # Sleep for a bit to ensure no overlapping measurements
    time.sleep(1) # adjust according to your needs

    # Save dataset every 1000 measurements
    if len(data) % 1000 == 0:
        df = pd.DataFrame(data)
        df.to_csv('dataset.csv', index=False)
