import cv2
import numpy as np
import imutils
from opencv_utils import get_bbox

class CircleDetect:
    def __init__(self, cam_device=0):
        self.minr_conf = 50
        self.maxr_conf = 400
        self.mind_conf = 150
        self.param1 = 30
        self.param2 = 45
        
    def __del__(self):
        pass

    def transform(self, frame):
        minr_conf = max(self.minr_conf, 0)
        maxr_conf = min(self.maxr_conf, 1000)
        mind_conf = max(self.mind_conf, 1)
        param1 = min(self.param1, 500)
        param2 = min(self.param2, 500)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.medianBlur(gray, 5)

        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                     cv2.THRESH_BINARY, 11, 3.5)

        kernel = np.ones((2, 3), np.uint8)
        gray = cv2.erode(gray, kernel)
        gray = cv2.dilate(gray, kernel)

        # print(mind_conf, param1, param2, minr_conf, maxr_conf)
        # detect circles in the image
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, mind_conf, param1=param1, param2=param2,
                                   minRadius=minr_conf, maxRadius=maxr_conf)
        
        object_area = 0
        object_x = 0
        object_y = 0
        object_w = 0
        object_h = 0
        center_x = 0
        center_y = 0
        # ensure at least some circles were found
        if circles is not None:
            # print(circles.shape)
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                frame = cv2.circle(frame, (x, y), r, (255, 255, 255), 1)
                # frame = cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

                found_area = 3.14159 * r * r
                if object_area < found_area:
                    object_area = found_area
                    object_x = x-r
                    object_y = y-r
                    object_w = r*2
                    object_h = r*2
                    center_x = x
                    center_y = y
                
            frame = cv2.rectangle(frame, (object_x, object_y), (object_x + object_w, object_y + object_h),
                                  (0, 255, 0), 2)
            frame = cv2.circle(frame, (center_x, center_y), radius=4, color=(0, 0, 255), thickness=2)

        return frame, [object_x, object_y, object_x + object_w, object_y + object_h]
#         ret, jpeg = cv2.imencode('.jpg', frame)

#         self.label_html = 'The Center is {},{}'.format(center_x, center_y)
#         return jpeg.tobytes()