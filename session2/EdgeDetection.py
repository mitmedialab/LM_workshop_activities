import cv2
import numpy as np
from opencv_utils import get_bbox

class EdgeDetection:
    def __init__(self, cam_device=0):
        self.thrs1_conf = 100
        self.thrs2_conf = 200
        self.edge_bluring = 25
        self.apt_conf = 3

    def transform(self, frame):
        # frame = cv2.flip(frame, 1)

        thrs_range = range(1, 501)
        apt_range = range(3, 8, 2)
        blure_range = range(3, 99)

        thrs1_conf = self.thrs1_conf if self.thrs1_conf in thrs_range else 100
        thrs2_conf = self.thrs2_conf if self.thrs2_conf in thrs_range else 200
        if self.edge_bluring % 2 == 0:
            self.edge_bluring = self.edge_bluring - 1
        edge_bluring = self.edge_bluring if self.edge_bluring in blure_range else 7
            
        apt_conf = self.apt_conf if self.apt_conf in apt_range else 3
        
        # print(thrs1_conf, thrs2_conf)
        imgBlur = cv2.medianBlur(frame, edge_bluring)
        img_edge = cv2.Canny(imgBlur, thrs1_conf, thrs2_conf, apertureSize=apt_conf)
        newframe = np.dstack([img_edge, img_edge, img_edge])

        [object_area, object_x, object_y,object_w,object_h,center_x,center_y] = get_bbox(img_edge)

        newframe = cv2.rectangle(newframe, 
                                 (object_x, object_y), 
                                 (object_x + object_w, object_y + object_h),
                             (0, 255, 0), 2)
        newframe = cv2.circle(newframe, (center_x, center_y), radius=4, color=(0, 0, 255), thickness=2)

        return newframe, [object_x, object_y, object_x + object_w, object_y + object_h]
