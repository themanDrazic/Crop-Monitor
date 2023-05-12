import tkinter as tk
from tkinter import messagebox
import VL53L1X
import threading
import time

# Global variable to control recording
recording = False

# Create a VL53L1X object
tof = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)


def record_distance():
    global recording

    # Open and start the VL53L1X sensor
    tof.open()
    tof.start_ranging(1)  # Start ranging, 1 = Short Range, 2 = Medium Range, 3 = Long Range

    while True:
        if recording:
            # Get a distance measurement
            distance_in_mm = tof.get_distance()

            # Write distance to file
            with open('distance_data.txt', 'a') as f:
                f.write("Distance: {0}mm\n".format(distance_in_mm))
        
            # Sleep for 50ms (20 times per second)
            time.sleep(0.05)
        else:
            time.sleep(0.1)

    # Stop ranging
    tof.stop_ranging()

    # Close the sensor
    tof.close()


def start_recording():
    global recording
    recording = True


def stop_recording():
    global recording
    recording = False


def exit_program():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        stop_recording()
        root.destroy()


# Initialize tkinter
root = tk.Tk()

# Create Start button
start_button = tk.Button(root, text="Start", command=start_recording)
start_button.pack()

# Create Stop button
stop_button = tk.Button(root, text="Stop", command=stop_recording)
stop_button.pack()

# Create Exit button
exit_button = tk.Button(root, text="Exit", command=exit_program)
exit_button.pack()

# Start recording thread
recording_thread = threading.Thread(target=record_distance)
recording_thread.start()

# Run tkinter main loop
root.protocol("WM_DELETE_WINDOW", exit_program)
root.mainloop()
