
# 20 photos are needed minimum to train the model
# Each picture must only have one face in it
#make sure the photos reflect what the camera is going to see
#we don't really care about the size of the face or how much of the body
#shows in the photo, as long as the face is clear and visible

import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image


# get the folder where the training images are    
folder= "faces/Mariya"

#list all files inside the designated faces folder
files =os.listdir(folder)

#load the face detection model
face_model=YOLO("yolov8m-face.pt")  
face_recognizer=cv2.face.LBPHFaceRecognizer_create()  # create the face recognizer model

faces=[]
# load each file with openCV

for file in files:

    photo=cv2.imread(folder+"/"+file)
    #pass photo into the model one at a time
    face_result=face_model(photo, verbose=False)# this returns their exact lcoation and some other stuff
    #process the image, meaning drawing a box over the face and displaying the result
    processed_image=face_result[0].plot()
    #gather the coodinates of the face detected in the photo
    left, top, right, bottom=face_result[0].boxes.xyxy[0].int()
    
    #we use 0 for the one and only face in the photo
    # we use into , as there are no decimal points in pixels, and we need to convert the float values to integers
    face=photo[top:bottom, left:right]
    #store  each face in a list, so we can use it later.
    faces.append(face)



#print what lives inside the face_result[0]
print(f'looking inside the face result: {face_result[0]}')
print(f'flooking inside the boxes:{face_result[0].boxes}')
print(f'flooking inside the boxes coordinates xyxy  :{face_result[0].boxes.xyxy}')




#test code to verify images are being read correctly
Image.fromarray(photo[:,:,::-1]).show()

Image.fromarray(processed_image[:,:,::-1]).show()
Image.fromarray(face[:,:,::-1]).show()


