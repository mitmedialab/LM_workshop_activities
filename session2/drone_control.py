import math
from djitellopy import Tello
import logging

# Speed of the drone
S = 60
# Maximum speed sent to send_rc_control
MAX_SPEED = 20

# if the distance to the target is less than the minimum then just
# set to zero to keep Tello close
MIN_DISTANCE = 10

SKIP_FRAME = 1

class DroneControl(object):
    def __init__(self, ip, cam='droneF'):
        # Init Tello object that interacts with the Tello drone
        self.tello = Tello(host=ip)
        self.tello.LOGGER.setLevel(logging.ERROR)
        self.cam = cam

        print('tello.init')
        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10
        self.centerX = 0
        self.centerY = 0
        
        self.send_rc_control = False
        
        try:
            self.tello.connect()
            self.tello.set_speed(self.speed)

            # In case streaming is on. This happens when we quit this program without the escape key.
            self.tello.streamoff()
            self.tello.streamon()
            if cam == 'droneD':
                self.tello.set_video_direction(Tello.CAMERA_DOWNWARD)
            else:
                self.tello.set_video_direction(Tello.CAMERA_FORWARD)

            self.frame_read = self.tello.get_frame_read()
            
            print('all.init')
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            print('Tello issues')

    def switch_camera(self, cam):
        if self.cam == cam:
            return
        self.cam = cam
        self.tello.streamoff()
        self.tello.streamon()
        if cam == 'droneD':
            self.tello.set_video_direction(Tello.CAMERA_DOWNWARD)
        else:
            self.tello.set_video_direction(Tello.CAMERA_FORWARD)

        return 
        
    def get_distances(self, coord):
        u_d_speed = 0
        l_r_speed = 0
        
        if self.tello:# and self.send_rc_control:
            x_distance = coord[0] - self.centerX
            y_distance = self.centerY - coord[1]
            
            distance = math.sqrt(x_distance ** 2 + y_distance ** 2)
            
            l_r_speed = int(((MAX_SPEED * x_distance) / self.centerX))
            # *-1 because the documentation says
            # that negative numbers go up but I am
            # seeing negative numbers go down
            u_d_speed = int((MAX_SPEED * y_distance / self.centerY))
            

            if abs(distance) <= MIN_DISTANCE:
                u_d_speed = 0
                l_r_speed = 0
                
        return l_r_speed, u_d_speed

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_t:  # takeoff
            self.takeoff()
        elif key == pygame.K_l:  # land
            not self.land()

    def keyup(self, key):
        pass
    
    def takeoff(self):
        print('taking off')
        self.tello.takeoff()
        self.send_rc_control = True
        
    def land(self):
        print('landing')
        self.tello.land()
        self.send_rc_control = False
        
    def follow_in_place(self, box_center_x, box_center_y):
        self.yaw_velocity = 0 
        self.for_back_velocity = 0
        self.left_right_velocity, self.up_down_velocity = self.get_distances((box_center_x, box_center_y))
        
    def follow_forward(self, box_center_x, box_center_y):
        self.yaw_velocity = 0 
        self.for_back_velocity = MIN_DISTANCE
        self.left_right_velocity, self.up_down_velocity = self.get_distances((box_center_x, box_center_y))
    def update(self):
        """ Update routine. Send velocities to Tello."""
        print(self.send_rc_control, 
              self.left_right_velocity, self.for_back_velocity,
              self.up_down_velocity, self.yaw_velocity)
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)

