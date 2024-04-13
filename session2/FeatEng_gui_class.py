import PySimpleGUI as sg
import cv2
import numpy as np
import sys
from icon import LMicon
from sys import platform
import os, json

from EdgeDetection import EdgeDetection
from ShapeDetection import ShapeDetection
from ColorDetection import ColorDetection
from circle_detection_class import CircleDetect
from drone_ip_lookup import *

ED = EdgeDetection()
ShD = ShapeDetection()
CD = ColorDetection()
CrD = CircleDetect()

working_dir = ''
if 'session2' in os.getcwd():
    working_dir = os.getcwd()
else:
    working_dir = os.path.join(os.getcwd(),'session2')

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = working_dir #os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def checkIntersection(boxA, boxB):
    
    x = max(boxA[0], boxB[0])
    y = max(boxA[1], boxB[1])
    w = min(boxA[0] + boxA[2], boxB[0] + boxB[2]) - x
    h = min(boxA[1] + boxA[3], boxB[1] + boxB[3]) - y

    foundIntersect = True
    if w < 0 or h < 0:
        foundIntersect = False

    return(foundIntersect, [x, y, w, h])

def main():
    width=640
    height=480
    fps = 30

    w_width=1300
    w_height=height+150
    
    p_width = int((656 - 2)/2) 
    p_image = 240

    select_cam = None
    prev_select_cam = None
    show_final = True
    show_edge = False
    show_thresh = False
    show_hsv = False
    show_circle = False

    #save HSV profile 
    HSV_profile_file = resource_path('HSV_profile.json')
    if os.path.exists(HSV_profile_file):
        with open(HSV_profile_file, 'r') as openfile:
            HSV_profile = json.load(openfile)
    else:
        HSV_profile = {
            "h_min": 0,
            "h_max": 179,
            "s_min": 0,
            "s_max": 255,
            "v_min": 0,
            "v_max": 255,
        }

    # drone actions  
    follow_forward = False
    drone_follow = False
    drone_stop = False
    
    def set_camera():
        global vidCam, DroneCtrl
        print('setting cam to:',select_cam)
        if select_cam == 'webcam':
            if platform == 'win32':
                vidCam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                vidCam = cv2.VideoCapture(0)
            vidCam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            vidCam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
            vidCam.set(cv2.CAP_PROP_FPS, fps)
        elif select_cam in ('droneF', 'droneD'):
            from drone_control import DroneControl
            if len(drone_ip) > 11:
                DroneCtrl = DroneControl(ip=drone_ip, cam=select_cam)
        

    set_camera()



    # ---===--- define the window layout --- #
    sg.theme('LightGray1')
    padding_text = ((5,5),(0,15))
        
    def hsv_row(title, s_range, key, value, title_top=True):
        slider_object = sg.Slider(range = s_range, 
                        orientation = 'h', 
                        enable_events = True, 
                        key=key,
                        default_value = value, 
                        size=(40, 10))
        title_object = sg.Text(title, font=("Helvetica", 14,'bold'))
        if title_top:
            return sg.Column([
            [title_object], #, size=(15, 1)
            [slider_object]
            ],vertical_alignment='top', element_justification='center')
        else:
            return sg.Column([
            [slider_object],
            [title_object], #, size=(15, 1)
            ],vertical_alignment='top', element_justification='center')

    
    #frames layout
    edge_frame_h = p_image + 100
    explaining_text = "Sensitivity thresholds filter out all weak edges, \n"
    explaining_text += "and connect the gaps in the detected edges. \n"
    explaining_image_process = "The biggest \'blob\' of white area is used \n to locate the object."
    edge_frame= [[
        sg.Column([
            [sg.Frame("",[
                [sg.Text(explaining_text, font=("Helvetica", 14))],
                [sg.Column([[]],size=(0,10))],
                [hsv_row('Low Sensitivity Threshold:',(1, 500), 'edge_th1', 100)],
                [hsv_row('High Sensitivity Threshold:',(1, 500), 'edge_th2', 200)],
            ],pad=((5,5),(5,5)), size=(350,170))],
            # [sg.Column([[]],size=(0,50))],
            [sg.Frame("",[
                [hsv_row('Smoothing:',(3, 99), 'edge_bluring', 25)], 
                [sg.Text("Smoothing blurs the image reducing unwanted edges.", font=("Helvetica", 14))]
            ],pad=((5,5),(5,5)))],
        ], element_justification='left',size=(p_width+50,edge_frame_h)),
        sg.Push(),
        sg.Column([
                [sg.Push(),sg.Text("Processed Frame", font=("Helvetica", 14, 'bold')),sg.Push()],
                [sg.Image(size=(p_image,p_image +20), pad=0, enable_events= True, k='-EDGE-')],
                [sg.Text(explaining_image_process, font=("Helvetica", 14))],
            ],vertical_alignment='top')
        ]]

    thresh_frame_h = p_image + 100
    explaining_text = "Threshold processing filters out pixel values beyond \n"
    explaining_text += "the min and max to locate objects. \n"
    thresh_frame= [[
            sg.Column([
                [sg.Frame("",[
                    [sg.Text(explaining_text, font=("Helvetica", 14))],
                    [hsv_row('Min:',(0, 255), 'thresh1', 50)],
                    [hsv_row('Max:',(0, 255), 'thresh2', 150)],
                    [sg.Column([[]],size=(0,10))],
                ],pad=((5,5),(5,5)))],
                [sg.Frame("",[
                    [hsv_row('Smoothing:',(3, 99), 'bluring', 25)],
                    [sg.Text("Smoothing blurs the image reducing diffrences \n in pixel values.", font=("Helvetica", 14))],
                ], pad=((5,5),(5,5)), size=(341,110))],
            ], element_justification='left',size=(p_width+50,thresh_frame_h)),
            sg.Push(),
        sg.Column([
                [sg.Push(),sg.Text("Processed Frame", font=("Helvetica", 14, 'bold')),sg.Push()],
                [sg.Image(size=(p_image,p_image +20), pad=0, enable_events= True, k='-THRESH-')],
                [sg.Text(explaining_image_process, font=("Helvetica", 14))],
            ],vertical_alignment='top')
        ]]

    hsv_frame_h = 2*p_image + 30
    hue_text = "Hue is the color appearance,\n representing how human brain process colors."
    sat_text = "Saturation (Chroma) is the strength of the hue.\n Faded colors have more gray values."
    val_text = "Value (lightness) describes the intensity\n of light or dark a color."
    hsv_frame= [
        ##change the photoshop image
        [sg.Column([
        [sg.Frame("",[
                [sg.Text(hue_text, font=("Helvetica", 14))],
                [hsv_row('min:',(0, 179), 'h_min', HSV_profile['h_min'])], 
                [sg.T('  0'),sg.Image(filename=resource_path('HSVmap_h.png'),
                size=(264,15), pad=0),sg.T('179')],
                [hsv_row('max:',(0, 179), 'h_max', HSV_profile['h_max'], False)]
            ],
            pad=((5,5),(5,5)), size=(343,145))],
        [sg.Frame("", [
                [sg.Text(sat_text, font=("Helvetica", 14))],
                [hsv_row('min:',(0, 255), 's_min', HSV_profile['s_min'])], 
                [sg.T('  0'),
                sg.Image(filename=resource_path('HSVmap_s.png'),size=(264,15), pad=0),
                sg.T('255')],
                [hsv_row('max:',(0, 255), 's_max', HSV_profile['s_max'], False)]],
            pad=((5,5),(5,5)), size=(343,145))],
         [sg.Frame("", [
                [sg.Text(val_text, font=("Helvetica", 14))],
                [hsv_row('min:',(0, 255), 'v_min', HSV_profile['v_min'])], 
                [sg.T('  0'),
                sg.Image(filename=resource_path('HSVmap_v.png'),size=(264,15), pad=0),
                sg.T('255')],
                [hsv_row('max:',(0, 255), 'v_max', HSV_profile['v_max'], False)]],
            pad=((5,5),(5,5)), size=(343,145))],
        ],
        vertical_alignment='top', size=(350, 750)),
        sg.Push(),
        sg.Column([
                [sg.Push(),sg.Text("Processed Frame", font=("Helvetica", 14, 'bold')),sg.Push()],
                [sg.Image(size=(p_image,p_image +20), pad=0, enable_events= True, k='-HSV-')],
                [sg.Text(explaining_image_process, font=("Helvetica", 14))],
            ],vertical_alignment='top')],
    ]
    
    circle_frame_h = 2*p_image + 30
    circle_frame= [[
        sg.Column([
            [sg.Frame("", [
                        [sg.Text("Min & Max Radius: controls the size of the detected\n circles", font=("Helvetica", 14))],
                        [hsv_row('min:',(0, width), 'minr_conf', 50)],
                        [hsv_row('max:',(0, width), 'maxr_conf', 200)]
                    ],pad=((5,5),(5,5)), size=(358,145))],
            [sg.Frame("",[
                        [hsv_row('Circles Distance:',(1, width), 'mind_conf', 150)],
                        [sg.Text("Controls the min distance between the detected circles.", font=("Helvetica", 14))],
                        [sg.Column([[]],size=(0,10))],
                        [hsv_row('Edge Threshold:',(1, 500), 'param1', 30)],
                        [sg.Text("Filters the weak edges bellow the threshold.", font=("Helvetica", 14))],
                        [sg.Column([[]],size=(0,10))],
                        [hsv_row('Center Threshold:',(1, 500), 'param2', 45)],
                        [sg.Text("Controls the circle\'s color distribution from its center.", font=("Helvetica", 14))],
                        
                    ],pad=((5,5),(5,5)))], 
            ], pad=((5,5),(5,5)), vertical_alignment='top'),
        sg.Push(),
        sg.Column([
                [sg.Push(),sg.Text("Processed Frame", font=("Helvetica", 14, 'bold')),sg.Push()],
                [sg.Image(size=(p_image,p_image +20), pad=0, enable_events= True, k='-CIRCLE-')],
                [sg.Text(explaining_image_process, font=("Helvetica", 14))],
            ],vertical_alignment='top')
        ]]
    
    def button_image(key, tip):
        return sg.Image(filename=resource_path('icon/'+key+'.png'),enable_events= True, k=key, background_color='#dfe0e2', tooltip = tip)#, size=(30,30))

    drone_frame_h = 85
    drone_frame = [
        [sg.Text("Choose the drone action given the detected bounding box",background_color='#dfe0e2', font=("Helvetica", 14))],
        [sg.Push(background_color='#dfe0e2'),
         button_image('drone_takeoff', 'Take off'),
         sg.Push(background_color='#dfe0e2'),
        #  button_image('drone_follow', 'Track target in place'), 
        #  sg.Push(background_color='#dfe0e2'),
         button_image('drone_stop', 'Stop tracking'),
         sg.Push(background_color='#dfe0e2'),
         button_image('drone_forward', 'Go to target!'),
         sg.Push(background_color='#dfe0e2'),
         button_image('drone_land', 'Land'),
         sg.Push(background_color='#dfe0e2'),
         sg.Quit(font=16,size=6)],
        ]
    

    camSel_frame_h = 50
    camSel_frame = [
        [sg.Text("Choose a camera:", font=("Helvetica", 14)),
         sg.Radio('WebCam', "camSelRadio", 
                  key= 'webcam_select', disabled=False, enable_events= True, font=("Helvetica", 14)), 
         sg.Radio('Drone Forward Cam', "camSelRadio", 
                  key= 'droneF_select', disabled=False, enable_events= True, font=("Helvetica", 14)),
         sg.Radio('Drone Downward Cam', "camSelRadio", 
                  key= 'droneD_select', disabled=False, enable_events= True, font=("Helvetica", 14))]]
    cam_selection = sg.Frame("Camera Selection", camSel_frame, 
                        size=(2*p_width-20, camSel_frame_h),pad=((5,5),(5,15)),vertical_alignment='top', font=("Helvetica", 16,'bold'))
    

     # ---===--- Sharifa please add one more tab with the original layout --- #
    explain_txt = 'Select the features that are suitable for the object you\n want to localize. \n\n'
    explain_txt += 'You might need different combination for each object.'
    explaining_fuse_process = "The final object location is the overlap\n between the bounding boxes of\n the selected features"
    select_features = [
        [sg.Column([
            [sg.Frame("",[[sg.T(explain_txt, font=("Helvetica", 14))],
                [sg.Column([[]],size=(0,10))],
                [sg.Checkbox('Edge Detection', False, key='cb_edge', enable_events=True,
                    visible=True, font=("Helvetica", 14))],
                [sg.Checkbox('Threshold Proccessing', False, key='cb_thresh', enable_events=True,
                    visible=True, font=("Helvetica", 14))],
                [sg.Checkbox('Hue Processing', False, key='cb_hsv', enable_events=True,
                    visible=True, font=("Helvetica", 14))],
                [sg.Checkbox('Circle Detection', False, key='cb_circle', enable_events=True,
                    visible=True, font=("Helvetica", 14))],
                [sg.Column([[]],size=(0,10))],
                ],pad=((5,5),(5,5)))],
            ],vertical_alignment='top',pad=((5,5),(5,15))),
        sg.Push(),
        sg.Column([
                [sg.Push(),sg.Text("Processed Frame", font=("Helvetica", 14, 'bold')),sg.Push()],
                [sg.Image(size=(p_image,p_image +20), pad=0, enable_events= True, k='-FUSE-')],
                [sg.Text(explaining_fuse_process, font=("Helvetica", 14))],
            ],vertical_alignment='top')
        ]]

    # ---===--- set tabs --- #

    tabgrp = [[cam_selection],
            [sg.TabGroup([[
                sg.Tab('Edge Detection', edge_frame),
                sg.Tab('Threshold Processing', thresh_frame),
                sg.Tab('Hue Processing', hsv_frame),
                sg.Tab('Circle Detection', circle_frame),
                sg.Tab('Select Features', select_features)
            ]], 
            font=("Helvetica", 14, 'bold'), tab_location='topleft',
            title_color='Black', tab_background_color='White',selected_title_color='Yellow',
            selected_background_color='Black', border_width=1, size=(2*p_width-20,470))
            ]
            ]  
    
    Ihead= sg.Image(filename=resource_path('FeatEng_header.png'),
                    size=(w_width,None),pad=0,enable_events= False, k='Ihead')
    Icol= sg.Image(filename=resource_path('FeatEng_columns.png'),
                   size=(w_width,None),pad=0,enable_events= False, k='Icol')
    Icam= sg.Image(size=(width,height),pad=0, enable_events= True, k='-CAM-')
#     dfe0e2
    layout= [[Ihead],
             # [Icol],
             [sg.Column(tabgrp,
                        size = (2*p_width-7, w_height+camSel_frame_h), 
                        vertical_alignment='top',
                        pad=((5,5),(5,5)),
                        scrollable = False,vertical_scroll_only = False,),
              sg.Column([[Icam],
                         [sg.Frame("Drone Action", drone_frame, 
                                   size=(2*p_width-30, drone_frame_h),
                                   pad=((5,5),(5,5)),background_color='#dfe0e2',
                                   vertical_alignment='top', font=("Helvetica", 16,'bold'))]
                        ],
                        size = (width, w_height), 
                        vertical_alignment='top',
                        background_color='#dfe0e2',
                        scrollable = False,vertical_scroll_only = False,),],
             ]

    # create the window and show it without the plot
    window = sg.Window('Seeing Machines', 
                       layout, 
                       # location=(0, 0),
                       no_titlebar=False, 
                       finalize=True, 
                       margins=(0, 0), 
                       element_padding=(0,0),
                       size = (w_width, w_height + drone_frame_h -20),
                       resizable=True,
                       icon=LMicon
                      )
    window.bind("<Escape>", "-ESCAPE-")
    # locate the elements we'll be updating. Does the search only 1 time
    image_elem = window['-CAM-']
    edges_elem = window['-EDGE-']
    thresh_elem = window['-THRESH-']
    colors_elem = window['-HSV-']
    circle_elem = window['-CIRCLE-']
    fuse_elem = window['-FUSE-']
    timeout = 1000//fps                 # time in ms to use for window reads



    # 
    def check_stream():
        if select_cam == 'webcam':
            return vidCam.isOpened() 
        else:
            return True
        
    def cleaning():
        if select_cam == 'webcam':
            vidCam.release()
        elif select_cam in ('droneF','droneD'): 
            DroneCtrl.tello.streamoff()
            DroneCtrl.tello.end()
        else:
            pass
            
    def switching():
        print('switching to:',select_cam)
        if select_cam == 'webcam':
            if 'DroneCtrl' in globals() or 'DroneCtrl' in locals():
                DroneCtrl.tello.streamoff()
                DroneCtrl.tello.end()
            set_camera()

        elif select_cam in ('droneF','droneD'): 
            if 'vidCam' in globals() or 'vidCam' in locals():
                vidCam.release()
            
            if 'DroneCtrl' in globals() or 'DroneCtrl' in locals():
                DroneCtrl.switch_camera(select_cam)
            else:
                print('init the drone')
                set_camera()

    drone_ip = '192.168.41.'
    while check_stream():
        event, values = window.read(timeout=timeout)
        # print(event)
        if event in (sg.WIN_CLOSED, sg.WINDOW_CLOSED,'Quit','-ESCAPE-', None):
            cleaning()
            break
        
        elif event in ('drone_takeoff') and select_cam in ('droneF','droneD'): 
            DroneCtrl.takeoff()
        elif event in ('drone_land') and select_cam in ('droneF','droneD'): 
            DroneCtrl.land()
        elif event in ('drone_forward'):  
            follow_forward = True
            drone_follow = False
            drone_stop = False
        elif event in ('drone_follow'):  
            follow_forward = False
            drone_follow = True
            drone_stop = False
        elif event in ('drone_stop'):  
            follow_forward = False
            drone_follow = False
            drone_stop = True

        elif event in ('-CAM-'):
            show_final = True
            show_edge = False
            show_thresh = False
            show_hsv = False
            show_circle = False
        elif event in ('-EDGE-'):
            show_edge = not show_edge
            show_final = True
            show_thresh = False
            show_hsv = False
            show_circle = False
        elif event in ('-THRESH-'):
            show_thresh = not show_thresh
            show_final = True
            show_edge = False
            show_hsv = False
            show_circle = False
        elif event in ('-HSV-'):
            show_hsv = not show_hsv
            show_final = True
            show_edge = False
            show_thresh = False
            show_circle = False
        elif event in ('-CIRCLE-'):
            show_circle = not show_circle
            show_final = True
            show_edge = False
            show_thresh = False
            show_hsv = False
            
        elif event in ('webcam_select', 'droneF_select', 'droneD_select'):
            # print('changing cam', values)
            prev_select_cam = select_cam
            if values['webcam_select']:
                select_cam = 'webcam'
            elif values['droneF_select']:
                select_cam = 'droneF'
                if len(drone_ip) <= 11 : #192.168.41.
                    drone_ip = popup_drone_ip(drone_ip)
                    print(drone_ip)
            elif values['droneD_select']:
                select_cam = 'droneD'
                if len(drone_ip) <= 11 : #192.168.41.
                    drone_ip = popup_drone_ip(drone_ip)
                    print(drone_ip)
            
            if prev_select_cam != select_cam:
                print('trying to switch cam')
                switching()
        elif event in ('h_min','h_max','v_min','v_max','s_min','s_max'):
            # save profile file
            HSV_profile = {
                "h_min": int(values['h_min']),
                "h_max": int(values['h_max']),
                "s_min": int(values['s_min']),
                "s_max": int(values['s_max']),
                "v_min": int(values['v_min']),
                "v_max": int(values['v_max']),
            }
            HSV_profile_object = json.dumps(HSV_profile, indent=4)
            with open(HSV_profile_file, "w") as outfile:
                outfile.write(HSV_profile_object)
               
        
        ED.thrs1_conf = int(values['edge_th1'])
        ED.thrs2_conf = int(values['edge_th2'])
        ED.edge_bluring = int(values['edge_bluring'])
        
        ShD.threshold1 = int(values['thresh1'])
        ShD.threshold2 = int(values['thresh2'])
        ShD.bluring = int(values['bluring'])
        
        CD.h_min = int(values['h_min'])
        CD.h_max = int(values['h_max'])
        CD.s_min = int(values['s_min'])
        CD.s_max = int(values['s_max'])
        CD.v_min = int(values['v_min'])
        CD.v_max = int(values['v_max'])
        
        CrD.minr_conf = int(values['minr_conf'])
        CrD.maxr_conf = int(values['maxr_conf'])
        CrD.mind_conf = int(values['mind_conf'])
        CrD.param1 = int(values['param1'])
        CrD.param2 = int(values['param2'])
        
        if select_cam == 'webcam':
            ret, frame = vidCam.read()
        elif select_cam  in ('droneF','droneD'):
            DroneCtrl.frame_read = DroneCtrl.tello.get_frame_read()
            frame = DroneCtrl.frame_read.frame
            ret = not DroneCtrl.frame_read.stopped
        else:
            frame = np.zeros([width,height,3],dtype=np.uint8)
            frame.fill(255)
            ret = True
            
            
        if not ret:  # if out of data stop looping
            break
        frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
        if select_cam  in ('droneF','droneD'):
            text = "Battery: {}%".format(DroneCtrl.tello.get_battery())
            frame = cv2.putText(frame, text, (5, height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            H, W, _ = frame.shape
            centerX = W // 2
            centerY = H // 2
            DroneCtrl.centerX = centerX
            DroneCtrl.centerY = centerY
        
        def process_feature(processed_frame):
            try:
                processed_frame = cv2.resize(processed_frame, (p_image, p_image), 
                                             interpolation = cv2.INTER_AREA)
                processed_imgbytes = cv2.imencode('.ppm', processed_frame)[1].tobytes()
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                processed_frame = np.zeros([p_image,p_image,3],dtype=np.uint8)
                processed_frame.fill(255)
                processed_imgbytes = cv2.imencode('.ppm', processed_frame)[1].tobytes()
            return processed_imgbytes
        
        
        edgeframe, edgebbox = ED.transform(frame.copy())
        edgesimgbytes = process_feature(edgeframe)
        edges_elem.update(data=edgesimgbytes)
        
        
        threshframe, threshbbox = ShD.transform(frame.copy())
        threshsimgbytes = process_feature(threshframe)
        thresh_elem.update(data=threshsimgbytes)

        colorframe, colorbbox = CD.transform(frame.copy())
        colorsimgbytes = process_feature(colorframe)
        colors_elem.update(data=colorsimgbytes)
        
        
        circelframe, circelbbox = CrD.transform(frame.copy())
        circelsimgbytes = process_feature(circelframe)
        circle_elem.update(data=circelsimgbytes)
        
        selectedBox = []
        if values['cb_edge']:
            selectedBox.append(edgebbox)
        if values['cb_thresh']:
            selectedBox.append(threshbbox)
        if values['cb_hsv']:
            selectedBox.append(colorbbox)
        if values['cb_circle']:
            selectedBox.append(circelbbox)
        
        fuse_frame = frame.copy()
        center_x, center_y = 0,0
        if len(selectedBox)>0:
            foundIntersect = False
            bbox = selectedBox[0]
            x, y, w, h = bbox
            fuse_frame = cv2.rectangle(fuse_frame, 
                                      (x, y), (w, h),
                                      (255, 0, 0), 2)
            for i in range(len(selectedBox)-1):
                box2 = selectedBox[i+1]
                x, y, w, h = box2
                fuse_frame = cv2.rectangle(fuse_frame, 
                                        (x, y), (w, h),
                                        (255, 0, 0), 2)
                foundIntersect, bbox = checkIntersection(bbox, box2)
                if not foundIntersect:
                    break

            if foundIntersect or len(selectedBox)==1:
                x, y, w, h = bbox
                center_x = int(x + ((w-x) / 2))
                center_y = int(y + ((h-y) / 2))
                frame = cv2.rectangle(frame, 
                                      (x, y),(w, h),
                                      (0, 255, 0), 2)
                frame = cv2.circle(frame, 
                                   (center_x, center_y), 
                                   radius=4, color=(0, 0, 255), thickness=2)
                fuse_frame = cv2.rectangle(fuse_frame, 
                                      (x, y), (w, h),
                                      (0, 255, 0), 2)
                fuse_frame = cv2.circle(fuse_frame, 
                                   (center_x, center_y), 
                                   radius=4, color=(0, 0, 255), thickness=2)
                
                H, W, _ = frame.shape
                centerX = W // 2
                centerY = H // 2
                if select_cam in ('droneF','droneB'):
                    frame = cv2.arrowedLine(frame, 
                                            (centerX, centerY), 
                                            (center_x, center_y),
                                    color=(0, 255, 0), thickness=2)
        
        if show_edge:
            final = edgeframe.copy()
        elif show_thresh:
            final = threshframe.copy()
        elif show_hsv:
            final = colorframe.copy()
        elif show_circle:
            final = circelframe.copy()
        else:
            final = frame.copy()
        imgbytes = cv2.imencode('.ppm', final)[1].tobytes()
        image_elem.update(data=imgbytes)


        fuseimgbytes = process_feature(fuse_frame)
        fuse_elem.update(data=fuseimgbytes)
        
       # 'drone_follow': True, 'drone_stop': False, 'drone_land': False
        if select_cam in ('droneF','droneB'):
            if drone_follow:
                if center_x > 0 and center_y > 0:
                    DroneCtrl.follow_in_place(center_x, center_y)
            elif drone_stop:
                DroneCtrl.left_right_velocity = 0
                DroneCtrl.for_back_velocity = 0
                DroneCtrl.up_down_velocity = 0 
                DroneCtrl.yaw_velocity = 0
            elif follow_forward:
                if center_x > 0 and center_y > 0:
                    DroneCtrl.follow_forward(center_x, center_y)
                
            DroneCtrl.update()
        
    window.close()

if __name__ == "__main__":
    main()