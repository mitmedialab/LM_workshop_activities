import cv2
import numpy as np
#import imutils
from opencv_utils import get_bbox

class ColorDetection:
    def __init__(self):
        # previous defaults 0 179 0 255 0 255
        self.h_min = 0
        self.h_max = 255
        self.s_min = 0
        self.s_max = 255
        self.v_min = 0
        self.v_max = 255

        self.label_html = ''
        self.bbox = ''

    def __del__(self):
        pass

    def set_from_form(self, form):
        self.h_min = form.get('h_min', default=self.h_min, type=int)
        self.h_max = form.get('h_max', default=self.h_max, type=int)
        self.s_min = form.get('s_min', default=self.s_min, type=int)
        self.s_max = form.get('s_max', default=self.s_max, type=int)
        self.v_min = form.get('v_min', default=self.v_min, type=int)
        self.v_max = form.get('v_max', default=self.v_max, type=int)

    def transform(self, frame):
        # frame = cv2.flip(frame, 1)

        #frame = imutils.resize(frame, width=650)
        #frame = cv2.flip(frame, 1)

        imgHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # h_min = cv2.getTrackbarPos("HUE (Min)","HSV")
        # h_max = cv2.getTrackbarPos("HUE (Max)", "HSV")
        # s_min = cv2.getTrackbarPos("SAT (Min)", "HSV")
        # s_max = cv2.getTrackbarPos("SAT (Max)", "HSV")
        # v_min = cv2.getTrackbarPos("VALUE (Min)", "HSV")
        # v_max = cv2.getTrackbarPos("VALUE (Max)", "HSV")
        # print(self.h_min, self.h_max, self.s_min,
        #      self.s_max, self.v_min, self.v_max)

        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])
        mask = cv2.inRange(imgHsv, lower, upper)
        
        [object_area, object_x, object_y, object_w, object_h, center_x, center_y] = get_bbox(mask)
        
        # result = cv2.bitwise_and(frame, frame, mask=mask)
        frame = np.dstack([mask,mask,mask])
        # frame = cv2.addWeighted(frame, 0.5, result, 0.5, 0)
        frame = cv2.rectangle(frame, (object_x, object_y), (object_x + object_w, object_y + object_h),
                              (0, 255, 0), 2)
        frame = cv2.circle(frame, (center_x, center_y), radius=4, color=(0, 0, 255), thickness=2)
        

        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # #hStack = np.hstack([frame,mask,result])
        # vStack = np.vstack([frame, mask, result])

        return frame, [object_x, object_y, object_x + object_w, object_y + object_h]
