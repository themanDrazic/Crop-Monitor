import time
import board
import busio
import adafruit_lidarlite

# Create library object using Bus I2C port
i2c = busio.I2C(board.SCL, board.SDA)

# Default configuration, with only i2c wires
sensor = adafruit_lidarlite.LIDARLite(i2c)

while True:
    try:
        # print the distance
        print((sensor.distance,))
    except RuntimeError as e:
        # If we get a reading error, print the type of the error
        print(e)
    time.sleep(0.2)
