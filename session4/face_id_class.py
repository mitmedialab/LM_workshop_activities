# from base64 import b64decode, b64encode
import cv2
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn import neighbors

import dlib
import math
import os

WHITE = (255, 255, 255)

set1_COLORS = [(102, 194, 165), (252, 141, 98), (141, 160, 203), (231, 138, 195), (166, 216, 84), (255, 217, 47), (229, 196, 148), (179, 179, 179)]
gray_color = (128,128,128)

def create_collages(images_list, required_size):
    n = len(images_list)

    empty_image = np.zeros([required_size[0],required_size[1],3],dtype=np.uint8)
    empty_image.fill(255)

    if n ==0:
        return empty_image
    if n == 1:
        # return 
        return np.hstack([images_list[0], empty_image, empty_image])

    nrows = int(math.sqrt(n) - 0.001) + 1    # best square fit for the given number
    ncols = max(int(math.ceil(n / nrows)),3)
    # print('Toatal face images: {}, which will be on {} rows and {} cols'.format(str(n),str(nrows),str(ncols)))
    im_c = 0 # counter for images 
    for this_raw in range(nrows):
        for this_col in range(ncols):
            if im_c < n:
                input_image = images_list[im_c].copy()
                im_c+=1
            else:
                input_image = empty_image

                # print(input_image.shape)
            cur_row = input_image if this_col == 0 else np.hstack([cur_row, input_image])
        collage = cur_row if this_raw == 0 else np.vstack([collage, cur_row])
    return collage
 

class FaceDetVis:
    def __init__(self, working_dir):
        face_detector='sfd'
        self.working_dir = working_dir
        # device='cuda'
        device='cpu'
        verbose=False
        face_detector_module = __import__('face_alignment.detection.' + face_detector, globals(), locals(), [face_detector], 0)
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose)
        self.confidence_threshold=0.75

        np.set_printoptions(suppress=False)
        path = os.path.join(working_dir,'models', 'dlib_face_recognition_resnet_model_v1.dat')
        self.model = dlib.face_recognition_model_v1(path)

        self.faces_array = []
        self.faces_encoding = []

        self.known_face_encodings = []
        self.known_face_names = []

        self.added_faces_images =[]
        self.added_faces_embds = []


        # Create and train the KNN classifier
        n_neighbors = 1
        knn_algo='ball_tree'
        self.distance_threshold=0.4
        self.limit_samples = 50
        self.knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')


        # print('*_*_*_*_**_*_*_*_* face init DONE _*_*_*_*_*_*_*_*')

    def update(self, image):
        pass

    def set_from_form(self, form):
        self.add_face_flag = form.get('add_flag', default='false', type=str)
        self.train_faces_flag = form.get('train_flag', default='false', type=str)
        self.face_name = form.get('face_name', default='Someone', type=str)

        if self.add_face_flag == 'true':
                self.add_a_face()
        elif self.train_faces_flag == 'true':
                self.face_training(self.face_name, True)

    def transform(self, image):
        # image = np.array(image)[:, :, ::-1].copy() 
        self.faces_array, faces_boxes = self.extract_face(image)
        if len(faces_boxes) < 1:
            return '' #image
        self.faces_encoding = [self.get_embedding(face_array).tolist() for face_array in self.faces_array]

        # print(np.unique(np.array(self.known_face_names)))
        names_encod = self.match_faces(self.faces_encoding, faces_boxes)

        num_face=-1
        # bbox_array = np.zeros([480,640,4], dtype=np.uint8)
        bbox_array = image.copy()
        # for (top, right, bottom, left, confidence), (face_name, dis) in zip(faces_boxes, names_encod):
        for (left, top, right, bottom), (face_name, dis) in zip(faces_boxes, names_encod):
            num_face +=1
            if num_face >= len(set1_COLORS):
                num_face = 0 
            color = set1_COLORS[num_face] if face_name != '?' else gray_color
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Draw a box around the face
            bbox_array = cv2.rectangle(bbox_array, (left, top), (right, bottom), color, 1)

            # Draw a label with a name below the face
            bbox_array = cv2.rectangle(bbox_array, (left, top - 15), (right-30, top), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            # text = '%s (%.1f)' % (face_name, dis)
            text = '%s' % (face_name)
            bbox_array = cv2.putText(bbox_array, text, (left + 6, top), font, 0.5, WHITE, 1)

        # bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 255
        return bbox_array #image

    # get the face embedding for one face
    def get_embedding(self, face_pixels):
        # scale pixel values
        if face_pixels is None:
            return

        yhat = np.array(self.model.compute_face_descriptor(face_pixels, num_jitters=0))
        return yhat
    

    def extract_face(self, image, required_size=(150, 150)):
        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        rgb_image = small_frame[:, :, ::-1]
        det_faces = self.face_detector.detect_from_image(rgb_image.copy())

        faces_array = []
        faces_boxes = []
        # ensure at least one face was found
        for i in range(0, len(det_faces)):
            confidence = det_faces[i][4]
            if confidence > self.confidence_threshold:
                box = det_faces[i][0:4]
                (startX, startY, endX, endY) = box.astype("int")

                face = rgb_image[startY:endY, startX:endX]
                try:
                    face_array = cv2.resize(face, required_size)
                    faces_array.append(face_array)
                    # faces_boxes.append((startY, endX, endY, startX, confidence))
                    faces_boxes.append((startX, startY, endX, endY))
                except:
                    pass

        return faces_array, faces_boxes

    def match_faces(self, faces_encoding, faces_boxes):
        if len(self.known_face_encodings) < 1:
            return [("?", 0) for encoding in faces_encoding]

        # self.knn_clf.fit(self.known_face_encodings,self.known_face_names)
        try:
            # TODO: fit at the start only

            closest_distances = self.knn_clf.kneighbors(faces_encoding, n_neighbors=1)

            distances = [closest_distances[0][i][0] for i in range(len(faces_boxes))]
            are_matches = [closest_distances[0][i][0] <= self.distance_threshold for i in range(len(faces_boxes))]
            names_encod = [(pred, dis) if rec else ("?", dis) \
                            for pred, encoding, rec, dis in 
                            zip(self.knn_clf.predict(faces_encoding), 
                                faces_encoding, are_matches, distances)]
        except Exception as e:
            #print(e)
            names_encod = [("?", 0) for encoding in faces_encoding]

        return names_encod

    def add_a_face(self):
        if len(self.faces_array) < 1 or len(self.faces_encoding) < 1 :
            return None, False

        self.added_faces_images.append(self.faces_array[0])

        self.added_faces_embds.append(self.faces_encoding[0])

        return self.show_face_collages(), False

    def show_face_collages(self):
        open_cv_image = create_collages(self.added_faces_images,required_size=(150, 150))
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        return open_cv_image

    def face_training(self, face_name, train_flag):
        #train the face, save it, model it, visulize it
        if len(self.added_faces_images) < 1 or len(self.added_faces_embds) < 1 or len(face_name) < 1:
            return None, train_flag

        if train_flag:
            name_list = [face_name for i in range(len(self.added_faces_embds))]
        
            self.known_face_encodings.extend(self.added_faces_embds)
            self.known_face_names.extend(name_list)

            # remove the temp list
            self.added_faces_embds = []
            self.added_faces_images = []

            try:
                # ic(type(self.known_face_encodings))
                self.knn_clf.fit(self.known_face_encodings, self.known_face_names)
            except Exception as e:
                #print(e)
                pass

            # print('*_*_*_*_*_*_*_* Training is Done *_*_*_*_*_*_*_* ')
            train_flag = False

        #return train_vis_chart, train_flag
        return train_flag
