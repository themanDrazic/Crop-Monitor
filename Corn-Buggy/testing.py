import time
import numpy as np
import cv2
from datetime import datetime
import RPi.GPIO as GPIO
import pigpio
if __name__ == '__main__':
	cap = cv2.VideoCapture(0)
	# set red thresh 
	lower_red = np.array([0,0,255])
	#156, 100, 40
	upper_red = np.array([180,255,255])
	while(1):
		ret, frame0 = cap.read()
		frame = cv2.flip(frame0,0)
		frame = frame[50:360,280:380]
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv, lower_red, upper_red)	
		edged = cv2.Canny(mask, 30, 200)	
		cv2.imshow('Canny Edges After Contouring', edged)
		_, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		print("Number of Contours found = " + str(len(contours)))
  
		# Draw all contours
		# -1 signifies drawing all contours
		# for c in contours:
		# 	M = cv2.moments(c)
		# 	cX = int(M["m10"] / M["m00"])
		# 	cY = int(M["m01"] / M["m00"])
		# 	cv2.drawContours(frame, c, -1, (0, 255, 0), 3)
		# 	cv2.circle(frame,(cX,cY),2,(255,255,255),-1)
		# 	cv2.putText(frame,"center",(cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
		cv2.imshow('Capture',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()

	cv2.destroyAllWindows()