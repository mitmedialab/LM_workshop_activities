import cv2
import socket

def get_bbox(color_mask):
	contours, hierachy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	object_area = 0
	object_x = 0
	object_y = 0
	object_w = 0
	object_h = 0
	center_x = 0
	center_y = 0
	
	for contour in contours:
		x, y, width, height = cv2.boundingRect(contour)
		found_area = width * height
		if object_area < found_area:
			object_area = found_area
			object_x = x
			object_y = y
			object_w = width
			object_h = height
			center_x = int(x + (width / 2))
			center_y = int(y + (height / 2))
	
	return [object_area, object_x, object_y, object_w, object_h, center_x, center_y]

def myVidCap(cam_device=0):
		# self.video = cv2.VideoCapture(cam_device)
		if cam_device == 'udp:/0.0.0.0:11111':
			tello_ip = '192.168.10.1'
			tello_port = 8889
			tello_address = (tello_ip, tello_port)
			tello_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			tello_socket.sendto('command'.encode(' utf-8 '), tello_address)
			tello_socket.sendto('streamon'.encode(' utf-8 '), tello_address)
			print("Start streaming")
			video = cv2.VideoCapture(cam_device, cv2.CAP_FFMPEG)
		else:
			video = cv2.VideoCapture(cam_device)
		return video