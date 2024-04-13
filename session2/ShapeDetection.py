import cv2
import numpy as np
#import imutils
from opencv_utils import get_bbox

#frameWidth = 640
#frameHeight = 480


class ShapeDetection:
    def __init__(self):
        # previous defaults 100 5000
        self.threshold1 = 50
        self.threshold2 = 150
        self.bluring = 25

    def transform(self, frame):
        if self.bluring % 2 == 0:
            self.bluring = self.bluring - 1
        # print(self.bluring,self.threshold1, self.threshold2)
        # imgBlur = cv2.GaussianBlur(frame, (self.bluring, self.bluring), 1)
        imgBlur = cv2.medianBlur(frame,self.bluring)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

        # th, imgCanny = cv2.threshold(
        #     imgGray, self.threshold1, 255, cv2.THRESH_BINARY)
        imgCanny = cv2.inRange(imgGray, self.threshold1, self.threshold2)
        
        newframe = np.dstack([imgCanny, imgCanny, imgCanny])
        
        [object_area, object_x, object_y,object_w,object_h,center_x,center_y] = get_bbox(imgCanny)
        
        newframe = cv2.rectangle(newframe, 
                                 (object_x, object_y), 
                                 (object_x + object_w, object_y + object_h),
                                 (0, 255, 0), 2)
        newframe = cv2.circle(newframe, 
                              (center_x, center_y), 
                              radius=4, color=(0, 0, 255), thickness=2)

        return newframe, [object_x, object_y, object_x + object_w, object_y + object_h]
