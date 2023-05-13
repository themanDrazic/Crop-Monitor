# Contains efficiency changes and improvements for code. Call these functions from 
# the original corn_buggy.py code in place of the less efficient ones there.
# might need a bit of debugging, but the functions all work individually.
import numpy as np
import cv2
import csv
from multiprocessing import Process, Queue
from datetime import datetime
import sys, pygame
from scipy.spatial import KDTree
max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
window_name = 'Threshold'
global pDict,thre_type,thre_value
#points_dict =  {1: [420, 476], 2: [408, 412], 3: [379, 385], 4: [360, 366], 5: [340, 347], 6: [323, 328], 7: [304, 310], 8: [286, 293], 9: [260, 264], 10: [241, 246], 11: [224, 229], 12: [205, 210], 13: [185, 191], 14: [164, 170], 15: [145, 153], 16: [126, 132], 17: [106, 112], 18: [85, 92], 19: [67, 72], 20: [23, 26]}
#points_dict = {1: [432, 476], 2: [411, 416], 3: [379, 388], 4: [358, 366], 5: [337, 346], 6: [318, 328], 7: [298, 308], 8: [280, 289], 9: [252, 260], 10: [233, 241], 11: [214, 224], 12: [196, 205], 13: [177, 185], 14: [155, 164], 15: [135, 145], 16: [117, 125], 17: [97, 105], 18: [76, 85], 19: [58, 65], 20: [4, 19]}
# create pressed button effect
def button(x2, y2, width, height, active_color, action = None): 
    cur = pygame.mouse.get_pos() 
    if x2 + width > cur[0] > x2 and y2 + height > cur[1] > y2: 
        s = pygame.Surface((width, height), pygame.SRCALPHA)   # per-pixel alpha transparent
        s.fill((161,255,255,128))                                # notice the alpha value in the color
        screen.blit(s, (x2,y2)) 
    else: 
        pass

# terminate the game
def terminate():
	pygame.quit()
	sys.exit()

def main():
	pygame.init()
	global black, bg,size, screen, twoDArray
	twoDArray = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
	mainPage = True
	startPage = False
	black = 0, 0, 0
	bg = pygame.image.load("field.jpeg")
	size = width, height = 320, 240
	screen = pygame.display.set_mode(size)
	while True:
		if mainPage:
			main_Page()
		elif startPage:
			start_Page()
		for event in pygame.event.get():
			# when left click
			if event.type == pygame.MOUSEBUTTONUP:
				# get the location when pressing the touchscreen
				pos = pygame.mouse.get_pos()
				x,y = pos
				# in start page
				if mainPage:
					if y < 155 and y > 130:
						if x < 110 and x > 30:
							print('Calibrate')
							cvAlgorithm(calibration=True)
						if x < 270 and x > 170:
							print('Start')
							startPage = True
							mainPage = False
							measuring()
			if event.type == pygame.QUIT:
				if mainPage:
					terminate()
				if startPage:
					startPage = False
					mainPage = True
def main_Page():
		screen.fill(black)
		screen.blit(bg,(0,0))
		button(30,130,100,25,[161,255,255])   # play button
		button(170,130,100,25,[161,255,255])   # score board button
		my_font = pygame.font.SysFont('timesnewroman',18)
		text_surface = my_font.render("Calibrate", True,black)
		rect = text_surface.get_rect(center = (80, 142))
		screen.blit(text_surface, rect) 
		my_font = pygame.font.SysFont('timesnewroman',18)
		text_surface = my_font.render("Start", True,black)
		rect = text_surface.get_rect(center = (220, 142))
		screen.blit(text_surface, rect) 
		pygame.display.flip()

def start_Page():
	screen.fill(black)
	for i,oneDArray in enumerate(twoDArray):
		for j, n in enumerate(oneDArray):
			if n == 0:
				color = (255,0,0)
			elif n == 1:
				color = (0,255,0)
			pygame.draw.circle(screen,color,(8+12*i,234-12*j),5)
	pygame.display.flip()



def onMouse(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDOWN:
		print('x = %d,y = %d'%(x,y))

def findContours(frame):
	contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	return contours, hierarchy 

def measuring():
	with open('parameter.csv','r') as csvfile:
		pDict = dict()
		reader = csv.reader(csvfile)
		for i,row in enumerate(reader):
                    print(row)
                    if (i<=18):
                        pDict[i+1] = [int(row[0]),int(row[1])]
                    elif i == 19:
                        thre_type = int(row[0])
                    else:
                        thre_value = int(row[0])
		print(pDict)
		print(thre_type)
		print(thre_value)
		cvAlgorithm(calibration=False,points_dict=pDict,thre_type=thre_type,thre_value=thre_value)



def write2CSV(frame,points_dict):
    #v = []
    #v = [0 for i in range(20)]
    v = {20:0,19:0,18:0,17:0,16:0,15:0,14:0,13:0,12:0,11:0,10:0,9:0,8:0,7:0,6:0,5:0,4:0,3:0,2:0,1:0}
    arr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    averY_dict = dict()
    with open('recorder5.csv','a+') as csvfile:
        now = datetime.now()
        current_time = now.strftime("%y:%m:%d:%H:%M:%S")
        writer = csv.writer(csvfile)
        contours, hierarchy = findContours(frame)
        #print("Number of Contours found = " + str(len(contours)))
        #writer.writerow([current_time,len(contours)])
        #points_dict = dict()
        if(len(contours) <=  19):
          for i,contour in enumerate(contours):
            temp_list = list()
            for c in contour:
                #averY = np.average(c[0][
                temp_list.append(c[0][1])
            averY = int(np.average(temp_list))
            averY_dict[19-i] = averY
            for key,value  in points_dict.items():
                #print([points_dict[key][0],points_dict[key][1]])
                if(averY <= (points_dict[key][1]+5) and averY > (points_dict[key][0]-5)):
                    v[key] = 1
                    arr[key-1] = 1
            twoDArray.pop()
            twoDArray.insert(0,arr)
        #print(averY)
          print(v)
          print(averY_dict)
        #print(points_dict)
        #writer.writerow([current_time,len(contours),v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15],v[16],v[17],v[18],v[19]])
          writer.writerow([current_time,len(contours),v,averY_dict])

def record_points_dict(frame):
	contours, hierarchy = findContours(frame)
	points_dict = dict()
	for i,contour in enumerate(contours):
		temp_list = list()
		for c in contour:
			temp_list.append(c[0][1])
		maxY = temp_list[np.argmax(temp_list)]
		minY = temp_list[np.argmin(temp_list)]
		#print(maxY)
		#print(minY)
		points_dict[i+1] = [minY,maxY]
	print(points_dict)
	return points_dict

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

from scipy.spatial import KDTree

def order_points_kdtree(pts):
    # Build a KDTree for efficient point operations
    kdtree = KDTree(pts)
    
    # Query the KDTree for sorted distances
    dists, idx = kdtree.query(pts, k=4)
    
    # Order points by ascending distance
    ordered_pts = pts[idx]

    return ordered_pts


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
