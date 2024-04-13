# from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time
import logging
import warnings
from .ipython_exit import ipy_exit

logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings('ignore')

class FrontEnd(object):
    FPS = 120

    def __init__(self, ip, model, class_to_find):
        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.width = 960
        self.height = 720
        self.screen = pygame.display.set_mode([self.width, self.height])

        # Init Tello object that interacts with the Tello drone
        # self.tello = Tello(host=ip)
        # self.tello.LOGGER.setLevel(logging.ERROR)

        
        self.cap = cv2.VideoCapture(0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap.set(cv2.CAP_PROP_FPS, 20)

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // self.FPS)
        
        self.model = model
        self.class_to_find = class_to_find

    def run(self):
        # self.tello.connect()

        # In case streaming is on. This happens when we quit this program without the escape key.
        # self.tello.streamoff()
        # self.tello.streamon()
        # frame_read = self.tello.get_frame_read()
        ok_frame, self.frame_read = self.cap.read() 

        should_stop = False
        while not should_stop:
            # if frame_read.stopped:
            #     break
            pygame.event.pump()
            ok_frame, self.frame_read = self.cap.read() 
            if not ok_frame:
                should_stop = True
                
                
            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

            self.screen.fill([0, 0, 0])

            # frame = frame_read.frame
            frame = self.frame_read
            frame = cv2.resize(frame, (self.width, self.height), interpolation = cv2.INTER_AREA)
            # text = "Battery: {}%".format(self.tello.get_battery())
            text = "Battery: {}%".format(100)
            frame = cv2.putText(frame, text, (5, self.height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.frame = frame
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictions = self.model.predict(frame)
            labels, boxes, scores = predictions
            
            if len(boxes) > 0:
                try:
                    obj_index = labels.index(self.class_to_find)
                    
                    box=boxes[obj_index]
                    x, y, w, h = int(box[0].item()), int(box[1].item()), \
                                int((box[2] - box[0]).item()), int((box[3] - box[1]).item())
                    box_center_x = int(x + (w / 2))
                    box_center_y = int(y + (h / 2))
                    
                    label = str(labels[obj_index])
                    # color = np.random.uniform(0, 255, size=(len(labels), 3))
                    color = (0, 0, 255)
                    font = cv2.FONT_HERSHEY_PLAIN

                    start_point, end_point = (x, y), (x + w, y + h)
                    frame = cv2.rectangle(frame, start_point, end_point , color, 2)
                    frame = cv2.putText(frame, '{}:{}'.format(label,str(scores[obj_index].item())),
                                      (x, y - 5), font, 2, color, 2)

                    frame = cv2.circle(frame, (box_center_x, box_center_y), radius=4, color=color,
                                     thickness=2)

                    H, W, _ = frame.shape
                    centerX = W // 2
                    centerY = H // 2
                    center_color = (255, 0, 255)

                    frame = cv2.circle(frame, center=(centerX, centerY), radius=5, color=center_color, thickness=-1)

                    frame = cv2.arrowedLine(frame, (centerX, centerY), (box_center_x, box_center_y),
                                          color=(0, 255, 0), thickness=2)
                except:
                    print('{} class does not exist in this frame.'.format(self.class_to_find))

            frame = np.rot90(frame)
            frame = np.flipud(frame)

            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / self.FPS)
        
        # Call it always before finishing. To deallocate resources.
        pygame.display.quit()
        pygame.quit()
        # self.tello.end()
        # sys.exit()
        exit = ipy_exit
        
        return 0
        
    def keydown(self, key):
        pass

    def keyup(self, key):
        pass

    def update(self):
        pass


