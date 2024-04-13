import cv2
import pygame
import numpy as np
import time, math
import torch
from detecto import core
# from djitellopy import Tello
import logging
import warnings

logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings('ignore')


#load camera calibration
from pathlib import Path

# Speed of the drone
S = 60
adding_move = 10
# Maximum speed sent to send_rc_control
MAX_SPEED = 20


# if the distance to the target is less than the minimum then just
# set to zero to keep Tello close
MIN_DISTANCE = 10
SPEED_FORWARD = 20

SKIP_FRAME = 1 
AI_SKIP_FRAMES = 3
MAX_MISSED = 30

class FrontEnd(object):
    FPS = 120
    def __init__(self, ip, selected_class = 'gate', conf_level = 0.75, class_file = 'predefined_classes.txt', model_file ='custom_model_weights.pth'):
        
        # classes_filepath = Path.home().joinpath( '.resources', class_file )
        classes_filepath = Path('./resources').joinpath(class_file)
        
        with open(classes_filepath) as f:
            classes_list = [line.rstrip() for line in f]

        self.model = core.Model(classes_list, model_name='fasterrcnn_mobilenet_v3_large_fpn')
        # save_custom_model_filepath = Path.home().joinpath( '.resources', model_file )
        save_custom_model_filepath = Path('./resources').joinpath(model_file)
        
        self.model.get_internal_model().load_state_dict(torch.load(save_custom_model_filepath, map_location=self.model._device))

        # Init pygame
        pygame.init()
        print('pygame.init')
        # Create pygame window
        pygame.display.set_caption("Tello video stream")
        self.width = 640
        self.height = 480
        self.screen = pygame.display.set_mode([self.width, self.height])
        self.find_obj_class = selected_class
        self.find_conf_level = conf_level
        self.frame_count = 0
       

        # Init Tello object that interacts with the Tello drone
        # self.data_queue=queue.Queue()
        # self.tello = Tello(host=ip)
        # self.tello.LOGGER.setLevel(logging.ERROR)
        
        self.cap = cv2.VideoCapture(0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap.set(cv2.CAP_PROP_FPS, 20)

        print('tello.init')
        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10
        
        self.centerX = self.width // 2
        self.centerY = self.height // 2
        
        self.class_id = 0
        self.selected_class = -1

        self.send_rc_control = False
       
        self.object_x = 0
        self.object_y = 0
        self.centerX = 0
        self.centerY = 0
        self.count = 0
        self.box = []
        self.label = ''
        self.score = 0
        self.object_found = False
        self.missing_object_count = 0
        
        
        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // self.FPS)

        print('all.init')

    def run(self):
        # self.tello.connect()
        # self.tello.set_speed(self.speed)

        # # In case streaming is on. This happens when we quit this program without the escape key.
        # self.tello.streamoff()
        # self.tello.streamon()

        # frame_read = self.tello.get_frame_read()

        
        ok_frame, self.frame_read = self.cap.read() 

        should_stop = False
        this_counter = 0
        
        while not should_stop:
            # if frame_read.stopped:
            #     break
            pygame.event.pump()
            ok_frame, self.frame_read = self.cap.read() 
            if not ok_frame:
                should_stop = True
                
                
            this_counter+=1
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

            # if frame_read.stopped:
            #     break
                        

            
            # frame = frame_read.frame
            frame = self.frame_read
            
            frame = cv2.resize(frame, (self.width, self.height), interpolation = cv2.INTER_AREA)
            # text = "Battery: {}%".format(self.tello.get_battery())
            text = "Battery: {}%".format(100)
            frame = cv2.putText(frame, text, (5, self.height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.frame = frame
            
            frame, [box_center_x, box_center_y] = self.find_object(self.frame.copy())
            if box_center_x > 0 and box_center_y > 0:
                        self.follow_forward(box_center_x, box_center_y)
            
            # preparing for showing in pygame
            self.screen.fill([0, 0, 0])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        # self.tello.land()
        
        return 0

    def find_object(self, frame):
        self.count += 1
        if self.count > AI_SKIP_FRAMES:
            self.count = 0
            predictions = self.model.predict(frame)
            labels, boxes, scores = predictions

            # Making bounding box with the arrow
            if len(boxes) <= 0:
                self.box = []
                return frame, [0,0]
            try:
                object_index = labels.index(self.find_obj_class)
            except:
                print('no object found')
                self.box = []
                return frame, [0,0]
            if scores[object_index].item() < self.find_conf_level:
                self.box = []
                return frame, [0,0]
                
            self.object_found = True
            self.missing_object_count = 0
            self.box=boxes[object_index]
            self.label = str(labels[object_index])
            self.score = scores[object_index].item()

        if len(self.box) <= 0:
            self.missing_object_count +=1
            self.scanning()
            return frame, [0,0]
        x, y, w, h = int(self.box[0].item()), int(self.box[1].item()), \
                    int((self.box[2] - self.box[0]).item()), int((self.box[3] - self.box[1]).item())
        box_center_x = int(x + (w / 2))
        box_center_y = int(y + (h / 2))

        
        color = (0, 0, 255)
        font = cv2.FONT_HERSHEY_PLAIN

        start_point, end_point = (x, y), (x + w, y + h)
        frame = cv2.rectangle(frame, start_point, end_point , color, 2)
        frame = cv2.putText(frame, '{}:{:.2f}'.format(self.label,self.score),
                            (x, y - 5), font, 2, color, 2)

        frame = cv2.circle(frame, (box_center_x, box_center_y), radius=4, color=color,
                            thickness=2)

        H, W, _ = frame.shape
        
        self.object_x = box_center_x
        self.object_y = box_center_y
        self.centerX = W // 2
        self.centerY = H // 2
        center_color = (255, 0, 255)

        frame = cv2.circle(frame, center=(self.centerX, self.centerY), radius=5, color=center_color, thickness=-1)

        frame = cv2.arrowedLine(frame, (self.centerX, self.centerY), (box_center_x, box_center_y),
                                color=(0, 255, 0), thickness=2)
        
        return frame, [box_center_x, box_center_y]

    def follow_forward(self, box_center_x, box_center_y):
        self.yaw_velocity = 0 
        self.for_back_velocity = SPEED_FORWARD
        self.left_right_velocity, self.up_down_velocity = self.get_distances((box_center_x, box_center_y))
    

    def get_distances(self, coord):
        u_d_speed = 0
        l_r_speed = 0
        
        # if self.tello:# and self.send_rc_control:
        #     x_distance = coord[0] - self.centerX
        #     y_distance = self.centerY - coord[1]
            
        #     distance = math.sqrt(x_distance ** 2 + y_distance ** 2)
            
        #     l_r_speed = int(((MAX_SPEED * x_distance) / self.centerX))
        #     # *-1 because the documentation says
        #     # that negative numbers go up but I am
        #     # seeing negative numbers go down
        #     u_d_speed = int((MAX_SPEED * y_distance / self.centerY))
            

        #     if abs(distance) <= MIN_DISTANCE:
        #         u_d_speed = 0
        #         l_r_speed = 0
                
        return l_r_speed, u_d_speed
        
    def scanning(self):
        if self.missing_object_count > MAX_MISSED:
            self.missing_object_count = 0
            self.object_found = False
            self.yaw_velocity = 30
            self.for_back_velocity = 0
            self.left_right_velocity = 0
            self.up_down_velocity = 0
        elif self.object_found: 
            self.yaw_velocity = 0
            self.for_back_velocity = SPEED_FORWARD
            self.left_right_velocity = 0
            self.up_down_velocity = 0
            
    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        pass
        if key == pygame.K_t:  # takeoff
            print('taking off')
            self.tello.takeoff()
            # self.homeKeyHit = False
            self.starFound = False
            self.homeArucoFound = False
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            print('landing')
            not self.tello.land()
            self.send_rc_control = False
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity += adding_move
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity += -adding_move
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity += -adding_move
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity += adding_move
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity += adding_move
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity += -adding_move
        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity += -adding_move
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity += adding_move
            

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        pass
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero up/down velocity -- make it stop
            self.up_down_velocity += 0
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity += 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity += 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity += 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity += 0

    def update(self):
        """ Update routine. Send velocities to Tello."""
        pass
        if self.send_rc_control:
            print(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)

