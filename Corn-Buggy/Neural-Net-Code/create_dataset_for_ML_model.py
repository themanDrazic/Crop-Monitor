import os
import cv2
import VL53L1X
import time
import requests
import json
import datetime
from PIL import Image

# Constants
OPEN_WEATHER_MAP_API_KEY = "your_openweathermap_api_key_here"
LATITUDE = "your_latitude"
LONGITUDE = "your_longitude"

# Function to get weather data
def get_weather_data():
    response = requests.get(f"http://api.openweathermap.org/data/2.5/weather?lat={LATITUDE}&lon={LONGITUDE}&appid={OPEN_WEATHER_MAP_API_KEY}")
    data = json.loads(response.text)
    return data['weather'][0]['main'], data['main']['temp']  # returns main weather and temperature

# Data Collection
csv_file = 'data.csv'
tof = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
tof.open()
cameras = get_connected_cameras()

with open(csv_file, 'w') as f:
    # Save dataset every 1000 measurements
    for _ in range(1000):  # adjust this as needed
        images = capture_images(cameras)
        tof.start_ranging(1)
        lidar_measurement = tof.get_distance()
        tof.stop_ranging()
        weather, temp = get_weather_data()
        time_of_day = datetime.datetime.now().strftime("%H:%M:%S")
        image_paths = [f'image_{i}_{time.time()}.jpg' for i in range(len(images))]
        for path, image in zip(image_paths, images):
            cv2.imwrite(path, image)
        f.write(','.join(image_paths) + ',' + str(lidar_measurement) + ',' + weather + ',' + str(temp) + ',' + time_of_day + '\n')
        time.sleep(5)

