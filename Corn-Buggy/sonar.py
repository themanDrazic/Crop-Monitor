import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
PIN = 23

GPIO.setup(PIN, GPIO.IN)

def measure():
    GPIO.wait_for_edge(PIN, GPIO.RISING)
    start_time = time.time()
    
    GPIO.wait_for_edge(PIN, GPIO.FALLING)
    end_time = time.time()
    
    duration = end_time - start_time
    distance = duration / 58e-6
    return distance

while True:
    distance = measure()
    print("Distance: {:.2f} cm".format(distance))
    time.sleep(1)

GPIO.cleanup()
