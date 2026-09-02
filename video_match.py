
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


# load each file with openCV

for file in files:

    photo=cv2.imread(folder+"/"+file)
    #pass photo into the model one at a time
    face_result=face_model(photo, verbose=False)# this returns their exact lcoation and some other stuff
    #process the image, meaning drawing a box over the face and displaying the result
    processed_image=face_result[0].plot()



#test code to verify images are being read correctly
Image.fromarray(photo[:,:,::-1]).show()
Image.fromarray(processed_image[:,:,::-1]).show()

