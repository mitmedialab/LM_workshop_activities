import PySimpleGUI as sg
import cv2
import sys
from face_id_class import FaceDetVis 
from icon import LMicon
from sys import platform
import os
import numpy as np

working_dir = ''
try:
    working_dir = sys._MEIPASS
except Exception:
    if 'session4' in os.getcwd():
        working_dir = os.getcwd()
    else:
        working_dir = os.path.join(os.getcwd(),'session4')

fd = FaceDetVis(working_dir)

def main():
    width=640
    height=480
    fps = 30
    
    w_width=1000
    w_height=height+210

    p_width = 357 - 10

    train_disabled = True

    if platform == 'win32':
        vidCam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        vidCam = cv2.VideoCapture(0)

    vidCam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    vidCam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
    vidCam.set(cv2.CAP_PROP_FPS, fps)

    # ---===--- define the window layout --- #
    sg.theme('LightGray1')
    padding_text = ((5,5),(0,15))
    Dlayout= [
        # [sg.T('Face Identification',font=("Helvetica", 32),pad=padding_text)],
              [sg.Frame('Step 1: Collect face images', 
                  [
                    [sg.Push(),sg.B('Add the current face', bind_return_key= True,key = '-AddFace-',font=16),sg.Push()],
                    [sg.T('Current faces in training sample:',font=("Helvetica", 12))],
                    [sg.Push(),sg.Image(enable_events= False, k='-Faces-', size = (300, 200)),sg.Push()]
                  ],
                  size=(p_width-10, 270),pad=((5,5),(0,5)),font=("Helvetica", 16,'bold'))],
              
              [sg.Frame('Step 2: Label the faces:', 
                  [
                    [sg.T('Name the person\'s face in the training sample:',font=("Helvetica", 12))],
                    [sg.Push(),sg.In(default_text ='Someone',s=20, font=18, k='face_name', pad=padding_text),sg.Push()],
                  ],
                  size=(p_width-10, 70),pad=((5,5),(0,5)),font=("Helvetica", 16,'bold'))],
              
              [sg.Frame('Step 3: Train the face ID model', 
                  [
                    [sg.Text('The samples in the models:', font=("Helvetica", 13,'bold')),
                    sg.Push(),
                    sg.B('    Train    ', bind_return_key= True,
                        key = '-Train-',font=16, pad=padding_text, disabled=train_disabled,
                        button_color = '#dfe0e2'),
                    sg.Push()],
                    [sg.Text(size=(40,90), font='Tahoma 13', key='-STLINE-'),]
                  ],
                  size=(p_width-10, 110),pad=((5,5),(0,5)),font=("Helvetica", 16,'bold'))],
              
              [sg.Frame('Step 4: Test the face ID model', 
                  [
                    [sg.T('Observe the demo space, does the model recognize you?',font=("Helvetica", 12))],
                    # [sg.T('',font=("Helvetica", 12))],
                  ],
                  size=(p_width-10, 50),pad=((5,5),(0,5)),font=("Helvetica", 16,'bold'))],
             ]

    
    Ihead= sg.Image(filename=os.path.join(working_dir,'faceID_header.png'),size=(w_width,None),pad=0,enable_events= False, k='Ihead')
    Icol= sg.Image(filename=os.path.join(working_dir,'faceID_columns.png'),size=(w_width,None),pad=0,enable_events= False, k='Icol')
    Icam= sg.Image(size=(width,height),pad=0, enable_events= True, k='-CAM-')
    layout= [[Ihead],
             [Icol],
             [sg.Column(Dlayout,
                        size = (p_width, w_height), 
                        vertical_alignment='top',
                        pad=((5,5),(5,5)),
                        scrollable = False,),
             sg.Column([[Icam],
                        [sg.Text('* Nor the images, nor the face models are stored. When you close the app, temporary memory is reset.',
                        font=("Helvetica", 12),pad=padding_text,background_color='#dfe0e2'),
                        sg.Push(background_color='#dfe0e2'),
                        sg.Quit(font=16,size=6)]],
                        size = (width, w_height), 
                        vertical_alignment='top',
                        scrollable = False,background_color='#dfe0e2')]]
    
    # create the window and show it without the plot
    window = sg.Window('Face Identification', 
                       layout, 
                       # location=(0, 0),
                       no_titlebar=False, 
                       finalize=True, 
                       margins=(0, 0), 
                       element_padding=(0,0),
                       size = (w_width, w_height),
                       resizable=True,
                       icon=LMicon
                      )
    window.bind("<Escape>", "-ESCAPE-")
    # locate the elements we'll be updating. Does the search only 1 time
    image_elem = window['-CAM-']
    faces_elem = window['-Faces-']
    train_elem = window['-Train-'] 
    sample_elem = window['-STLINE-'] 
    timeout = 1000//fps                 # time in ms to use for window reads
    
    def check_button():
        if len(fd.added_faces_images) > 0:
            train_elem.update(disabled=False, button_color = '#111111')
        else:
            train_elem.update(disabled=True, button_color = '#dfe0e2')

    def model_sample():
        face_names, face_counts = np.unique(fd.known_face_names, return_counts=True)
        text_samples = ''
        for name, count in zip(face_names, face_counts):
            text_samples += '{}: {}, '.format(name, count)

        sample_elem.update(text_samples)
    
    while vidCam.isOpened():
        event, values = window.read(timeout=timeout)
        if event in (sg.WIN_CLOSED, sg.WINDOW_CLOSED,'Quit','-ESCAPE-', None):
            vidCam.release()
            break
        elif event in ('-AddFace-'):
            fd.add_a_face()
            check_button()
        elif event in ('-Train-'):
            face_name = values['face_name']
            fd.face_training(face_name, True)
            check_button()
            model_sample()
            
        ret, frame = vidCam.read()
        if not ret:  # if out of data stop looping
            break
        frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
        bbframe = fd.transform(frame)

        try:
            imgbytes = cv2.imencode('.ppm', bbframe)[1].tobytes()  
        except:
            imgbytes = cv2.imencode('.ppm', frame)[1].tobytes()
        image_elem.update(data=imgbytes)
        
        faceCollage=fd.show_face_collages()
        faceCollage = cv2.resize(faceCollage, (300, 200), interpolation = cv2.INTER_AREA)
        faceimgbytes = cv2.imencode('.ppm', faceCollage)[1].tobytes()
        faces_elem.update(data=faceimgbytes)

    window.close()

main()