# Contains efficiency changes and improvements for code. Call these functions from 
# the original corn_buggy.py code in place of the less efficient ones there.
import numpy as np
import cv2
import csv
from multiprocessing import Process, Queue

def Threshold(calibration, g, window_name, points_dict=dict(), thre_type=3, thre_value=175):
    threshold_type = thre_type
    threshold_value = thre_value
    _, dst = cv2.threshold(g, threshold_value, max_value, threshold_type)
    dst = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    dim = (800, 800)
    resized = cv2.resize(dst, dim, interpolation=cv2.INTER_AREA)
    if calibration:
        points_dict = record_points_dict(resized)
        return points_dict, threshold_type, threshold_value
    else:
        write2CSV(resized, points_dict)

# All other functions like 'order_points', 'four_point_transform' remain same.

def capture_frames(queue):
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    while True:
        ret, frame0 = cap.read()
        if not ret:
            break
        queue.put(frame0)

def process_frames(queue, calibration=False, points_dict=dict(), thre_type=3, thre_value=175):
    if calibration:
        cv2.namedWindow(window_name)
        cv2.createTrackbar(trackbar_type, window_name , 3, max_type, Threshold)
        cv2.createTrackbar(trackbar_value, window_name , 0, max_value, Threshold)

    while True:
        frame0 = queue.get()
        frame = cv2.resize(frame0,(640,480))
        frame = cv2.flip(frame,0)
        src = np.array([[138, 15],[373, 26],[346, 449],[173, 470]], dtype = "float32")
        dst = np.array([[0, 0],[480, 0],[640, 480],[0, 640]], dtype = "float32")
        frame = four_point_transform(frame, src)
        g = frame[:,:,1]
        cv2.setMouseCallback('Image',onMouse)
        cv2.setMouseCallback('threshold',onMouse)

        if calibration:
            points_dict,threshold_type,threshold_value = Threshold(calibration, g, window_name, points_dict, thre_type, thre_value)
        else:
            Threshold(calibration, g, window_name, points_dict, thre_type, thre_value)
            start_Page()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if calibration:
                with open('parameter.csv','w') as parafile:
                    parawriter = csv.writer(parafile)
                    for value in points_dict.values():
                        parawriter.writerow(value)
                    parawriter.writerow([threshold_type])
                    parawriter.writerow([threshold_value])
                print(points_dict)
                print(threshold_type)
                print(threshold_value)
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
        frame_queue = Queue(maxsize=10)
        capture_process = Process(target=capture_frames, args=(frame_queue,))
        process_process = Process(target=process_frames, args=(frame_queue,))
        
        # start the processes
        capture_process.start()
        process_process.start()
        
        # wait for processes to complete
        capture_process.join()
        process_process.join()
