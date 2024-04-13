import cv2
import pygame
import numpy as np
import time, math
import torch
from detecto import core
from djitellopy import Tello
import logging
import warnings
from .ipython_exit import ipy_exit

logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings('ignore')


#load camera calibration
from pathlib import Path

# Speed of the drone
S = 60

# Maximum speed sent to send_rc_control
MAX_SPEED_X = 40
MAX_SPEED_Y = 50
# Speed for autonomous navigation
S_prog = 15

# if the distance to the target is less than the minimum then just
# set to zero to keep Tello close
MIN_DISTANCE = 50
MIN_HEIGHT= 5
HEIGHT_ERROR_MARGIN = 20

# Flag indicates if the main handler should send a command to stop
# all motion of the Tello
STOP_TELLO_MOTION = False

SKIP_FRAME = 0
#MAX frames missing a star after it's been found
MAX_MISSED_STAR_FRAMES = 5
MAX_STIL_NOT_STAR = 30


class FrontEnd(object):
    FPS = 120
    def __init__(self, ip, selected_class = 'star', conf_level = 0.75, class_file = 'predefined_classes.txt', model_file ='custom_model_weights.pth'):
        
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
        self.tello = Tello(host=ip)
        self.tello.LOGGER.setLevel(logging.ERROR)

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
        self.mission1 = False
        self.star_found = False
        self.missing_star_count = 0
        self.still_not_star = 0
       
        
        self.starGot = False
      
        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // self.FPS)

        print('all.init')

    def run(self):
        self.tello.connect()
        self.tello.set_speed(self.speed)

        # In case streaming is on. This happens when we quit this program without the escape key.
        self.tello.streamoff()
        self.tello.streamon()

        frame_read = self.tello.get_frame_read()

        should_stop = False
        this_counter = 0
        
        while not should_stop:
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

            if frame_read.stopped:
                break
                        

            
            frame = frame_read.frame
            frame = cv2.resize(frame, (self.width, self.height), interpolation = cv2.INTER_AREA)
            text = "Battery: {}%".format(self.tello.get_battery())
            frame = cv2.putText(frame, text, (5, self.height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.frame = frame
            
            
                    
            if not self.mission1:
                self.find_star()
          
        
            frame = self.frame    
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
        self.tello.end()
        # sys.exit()
        exit = ipy_exit
        
        return 0

    def find_star(self):
        frame = self.frame
        self.frame_count +=1
        if self.frame_count % SKIP_FRAME:
            return ''
        predictions = self.model.predict(frame)
        labels, boxes, scores = predictions

        # Making bounding box with the arrow
        if len(boxes) > 0:
            self.count = 0
            try:
                star_index = labels.index(self.find_obj_class)
            except:
                print('no object found, keep scanning')
                self.navigation(frame)
                
                return ''
            if scores[star_index].item() < self.find_conf_level:
                self.navigation(frame)
                return ''
            self.star_found = True
            box=boxes[star_index]
            x, y, w, h = int(box[0].item()), int(box[1].item()), \
                        int((box[2] - box[0]).item()), int((box[3] - box[1]).item())
            box_center_x = int(x + (w / 2))
            box_center_y = int(y + (h / 2))
            label = str(labels[star_index])
            color = (0, 0, 255)
            font = cv2.FONT_HERSHEY_PLAIN

            start_point, end_point = (x, y), (x + w, y + h)
            frame = cv2.rectangle(frame, start_point, end_point , color, 2)
            frame = cv2.putText(frame, '{}:{}'.format(label,str(scores[star_index].item())),
                              (x, y - 5), font, 2, color, 2)

            frame = cv2.circle(frame, (box_center_x, box_center_y), radius=4, color=color,
                             thickness=2)


            center_color = (255, 0, 255)

            frame = cv2.circle(frame, center=(self.centerX, self.centerY), radius=5, color=center_color, thickness=-1)

            frame = cv2.arrowedLine(frame, (self.centerX, self.centerY), (box_center_x, box_center_y),
                                  color=(0, 255, 0), thickness=2)
            
            
            print('_*_**_*_*_*_*_*_*__**__*_*_An object found _*_**_*_*_*_*_*_*__**__*_*_')
            print(self.centerX,self.centerY,box_center_x, box_center_y)
            
            self.yaw_velocity = 0 
            self.left_right_velocity, self.up_down_velocity, self.for_back_velocity = self.get_distances((box_center_x, box_center_y))
           
            self.frame = frame
            
        else:
            self.navigation(frame)
     

        return ''
        
    
    def get_distances(self, starCoord):
        u_d_speed = 0
        l_r_speed = 0
        f_b_speed = 0
        
        if self.tello and self.send_rc_control:
            x_distance = starCoord[0] - self.centerX
            y_distance = self.centerY - starCoord[1]
           
            
            distance = math.sqrt(x_distance ** 2 + y_distance ** 2)
            
            l_r_speed = int(((MAX_SPEED_X * x_distance) / self.centerX))
            u_d_speed = int((MAX_SPEED_Y * y_distance / self.centerY))
            

            print(l_r_speed,u_d_speed)
            f_b_speed = 30
            
            if abs(distance) <= 20:
                u_d_speed = 0
                l_r_speed = 0
                f_b_speed = 10
                
        return l_r_speed, u_d_speed, f_b_speed
    
    def navigation(self, frame):
        # self.scanning()
        if self.star_found:
            self.missing_star_count +=1
            if self.missing_star_count > MAX_MISSED_STAR_FRAMES:
                self.scanning()
                self.missing_star_count = 0
        else:
            self.scanning()
           
    def scanning(self):
            self.yaw_velocity = 30
            self.for_back_velocity = 0
            self.left_right_velocity = 0
            self.up_down_velocity = 0
            
            self.still_not_star += 1
            if self.still_not_star > MAX_STIL_NOT_STAR:
                self.up_down_velocity = 10
                self.for_back_velocity = 20
                self.still_not_star = 0

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
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
        elif key == pygame.K_UP:  # set move up/down velocity
            self.up_down_velocity = 10
        elif key == pygame.K_DOWN:
            self.up_down_velocity = -10
            

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero up/down velocity -- make it stop
            self.up_down_velocity = 0

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            print(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)

