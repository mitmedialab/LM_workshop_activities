# from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time, sys
from pathlib import Path

import logging
import warnings
from .ipython_exit import ipy_exit

logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings('ignore')

# Speed of the drone
S = 60
# Frames per second of the pygame window display
# A low number also results in input lag, as input information is processed once per frame.
FPS = 120


class FrontEnd(object):
    def __init__(self,ip):
        # Init pygame

        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Init Tello object that interacts with the Tello drone
        # self.tello = Tello(host=ip)
        #0 Is the built in camera
        self.cap = cv2.VideoCapture(0)
        #Gets fps of your camera
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # print("fps:", fps)
        #If your camera can achieve 60 fps
        #Else just have this be 1-30 fps
        self.cap.set(cv2.CAP_PROP_FPS, 20)

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

    def run(self):

        # self.tello.connect()
        # self.tello.set_speed(self.speed)

        # In case streaming is on. This happens when we quit this program without the escape key.
        # self.tello.streamoff()
        # self.tello.streamon()

        # frame_read = self.tello.get_frame_read()
        ok_frame, self.frame_read = self.cap.read() 
        
        should_stop = False
        while not should_stop:
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

            # if frame_read.stopped:
            #     should_stop = True

            self.screen.fill([0, 0, 0])

            # frame = frame_read.frame
            frame = self.frame_read

            # battery n. 
            # text = "Battery: {}%".format(self.tello.get_battery())
            text = "Battery: {}%".format(100)
            cv2.putText(frame, text, (5, 720 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = np.flipud(frame)

            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)

        # Call it always before finishing. To deallocate resources.

        print('we are getting out')
        # self.tello.end()
        pygame.display.quit()
        pygame.quit()
        
    def takeSnapshot(self):
        """
        save the current frame of the video as a jpg file and put it into outputpath
        """

        # grab the current timestamp and use it to construct the filename
        import datetime, os
        ts = datetime.datetime.now()
        outputPath = './images'
        
        try:
            os.makedirs(outputPath, exist_ok = True)
        except OSError as error:
            pass
  
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))

        p = os.path.sep.join((str(outputPath), filename))
        frame = self.frame_read #self.tello.get_frame_read().frame
        # save the file
        cv2.imwrite(p, frame) # cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print("[INFO] saved {}".format(p))
        
    def keydown(self, key):
        """ Update velocities based on key pressed
        """
        return ''
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = S

    def keyup(self, key):
        """ Update velocities based on key released     
        """
        # if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
        #     self.for_back_velocity = 0
        # elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
        #     self.left_right_velocity = 0
        # elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
        #     self.up_down_velocity = 0
        # elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
        #     self.yaw_velocity = 0
        # elif key == pygame.K_t:  # takeoff
        #     self.tello.takeoff()
        #     self.send_rc_control = True
        # elif key == pygame.K_l:  # land
        #     not self.tello.land()
        #     self.send_rc_control = False
        # elif key == pygame.K_SPACE:  # save image
        #     self.takeSnapshot()
        if key == pygame.K_SPACE:  # save image
            self.takeSnapshot()

    def update(self):
        """ Update routine. Send velocities to Tello.
        """
        return ''
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)

