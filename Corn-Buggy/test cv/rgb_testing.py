import cv2
import numpy as np
import csv
from datetime import datetime

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
window_name = 'Threshold'
#points_dict =  {1: [420, 476], 2: [408, 412], 3: [379, 385], 4: [360, 366], 5: [340, 347], 6: [323, 328], 7: [304, 310], 8: [286, 293], 9: [260, 264], 10: [241, 246], 11: [224, 229], 12: [205, 210], 13: [185, 191], 14: [164, 170], 15: [145, 153], 16: [126, 132], 17: [106, 112], 18: [85, 92], 19: [67, 72], 20: [23, 26]}
points_dict = {1: [432, 476], 2: [411, 416], 3: [379, 388], 4: [358, 366], 5: [337, 346], 6: [318, 328], 7: [298, 308], 8: [280, 289], 9: [252, 260], 10: [233, 241], 11: [214, 224], 12: [196, 205], 13: [177, 185], 14: [155, 164], 15: [135, 145], 16: [117, 125], 17: [97, 105], 18: [76, 85], 19: [58, 65], 20: [4, 19]}

def onMouse(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDOWN:
		print('x = %d,y = %d'%(x,y))

def findContours(frame):
    #v = []
    #v = [0 for i in range(20)]
    v = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0}
    averY_dict = dict()
    with open('recorder.csv','a+') as csvfile:
        now = datetime.now()
        current_time = now.strftime("%y:%m:%d:%H:%M:%S")
        writer = csv.writer(csvfile)
        contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print("Number of Contours found = " + str(len(contours)))
        #writer.writerow([current_time,len(contours)])
        #points_dict = dict()
        if(len(contours) <=  20):
          for i,contour in enumerate(contours):
            temp_list = list()
            for c in contour:
                #averY = np.average(c[0][
                temp_list.append(c[0][1])
            averY = np.average(temp_list)
            averY_dict[19-i] = int(averY)
            #maxY = temp_list[np.argmax(temp_list)]
            #minY = temp_list[np.argmin(temp_list)]
		#print(maxY)
		#print(minY)
            #points_dict[i+1] = [minY,maxY]
            #print(averY)
            for key,value  in points_dict.items():
                #print([points_dict[key][0],points_dict[key][1]])
                if(averY <= (points_dict[key][1]+4) and averY > (points_dict[key][0]-4)):
                    v[key] = 1
        #print(averY)
          print(v)
          print(averY_dict)
        #print(points_dict)
        #writer.writerow([current_time,len(contours),v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15],v[16],v[17],v[18],v[19]])
          writer.writerow([current_time,len(contours),v,averY_dict])


def Threshold(channel,window_name):
    #0: Binary
    #1: Binary Inverted
    #2: Threshold Truncated
    #3: Threshold to Zero
    #4: Threshold to Zero Inverted
    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)
    _, dst = cv2.threshold(channel, threshold_value, max_binary_value, threshold_type )
    findContours(dst)
    r = 250.0 / dst.shape[1]
    dim = (250, int(dst.shape[0] * r))
    resized = cv2.resize(dst, dim, interpolation=cv2.INTER_AREA)
    #frame1 = dst[:,:217]
    #frame2 = dst[:,217:340]
    #frame3 = dst[:,340:]
    #cv2.imshow('frame1',frame1)
    #cv2.imshow('frame2',frame2)
    #cv2.imshow('frame3',frame3)
    cv2.imshow(window_name,resized)

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(window_name)
    cv2.createTrackbar(trackbar_type, window_name , 3, max_type, Threshold)
    # Create Trackbar to choose Threshold value
    cv2.createTrackbar(trackbar_value, window_name , 0, max_value, Threshold)
    #cap.set(3,1920)
    #cap.set(4,1080)
    cap.set(3,1280)
    cap.set(4,720)
    while(1):
        ret, frame0 = cap.read()
        #cv2.imshow('frame0',frame0)
	#frame0 = cv2.imread('laser.jpg')
        frame = cv2.resize(frame0,(640,480))
        frame = cv2.flip(frame,0)
        cv2.imshow('frame',frame)
        src = np.array([
		[138, 15],
		[373, 26],
		[346, 449],
		[173, 470]], dtype = "float32")
        dst = np.array([
		[0, 0],
		[480, 0],
		[640, 480],
		[0, 640]], dtype = "float32")
		# M = cv2.getPerspectiveTransform(src,dst)
		# warped = cv2.warpPerspective(frame,M,(640,480))
        frame = four_point_transform(frame, src)
		#frame = frame[:,125:180]
        cv2.imshow('Image', frame)
        #frame4 = frame[]
        cv2.setMouseCallback('Image',onMouse)
	#cv2.imshow('Image', warped)
        b, g, r = cv2.split(frame)
        #cv2.imshow('g',g)
        cv2.setMouseCallback('threshold',onMouse)
	# Call the function to initialize
        Threshold(g,window_name)
        #_,dst = cv2.threshold(g,198,255,cv2.THRESH_BINARY)
        #r = 350.0/dst.shape[1]
        #dim = (350,int(dst.shape[0]*r))
        #resized = cv2.resize(dst,dim,interpolation=cv2.INTER_AREA)
        #frame1 = resized[:189,:]
        #frame2 = resized[189:272,:]
        #cv2.imshow('threshold',resized)
        #cv2.imshow('frame1',frame1)
        #cv2.imshow('frame2',frame2)
        #Threshold(g1,'g1')
		# Wait until user finishes program
		# ret,thresh1 = cv2.threshold(g, 225,255,cv2.THRESH_BINARY)
		# ret,thresh2 = cv2.threshold(g, 200,255,cv2.THRESH_BINARY_INV)
		# ret,thresh3 = cv2.threshold(g, 200,255,cv2.THRESH_TRUNC)
		# ret,thresh4 = cv2.threshold(g, 200,255,cv2.THRESH_TOZERO)
		# ret,thresh5 = cv2.threshold(g, 200,255,cv2.THRESH_TOZERO_INV)
		#_, contours, hierarchy = cv2.findContours(newIm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# cv2.imshow('New Image', thresh1)
		#print("Number of Contours found = " + str(len(contours)))
		# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
		# images = [frame,thresh1, thresh2, thresh3, thresh4, thresh5]
		# for i in range(6):
		#     cv2.imshow('image'+str(i),images[i])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
