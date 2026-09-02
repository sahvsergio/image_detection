
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
faces=[]
labels=[]


#load the face detection model
face_model=YOLO("yolov8m-face.pt")  
face_recognizer=cv2.face.LBPHFaceRecognizer_create()  # create the face recognizer model

# get the folder where the training images are    

folders={0:'faces/Mariya',1:'faces/Mario'} # if we had more than one person, we would add them here with a different number for each person
#iterate over the dictionary of folders, where the key is the label and the value is the folder path
for label, folder in folders.items():
    #list all files inside the designated faces folder
    files =os.listdir(folder)
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
        # # we use into , as there are no decimal points in pixels, and we need to convert the float values to integers
        face=photo[top:bottom, left:right]
        # turn into a gray color for the face recognizer to work with and reassign the face variable to the new gray image
        face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        #resize the face to a standard size, so the recognizer can work with it
        face=cv2.resize(face, (200,200))
        #store  each face in a list, so we can use it later.
        faces.append(face)
        labels.append(label)# we use label now since we pull it form the dictionary of labels

face_recognizer.train(faces,np.array(labels)) # train the recognizer with the faces and labels we have collected  
face_recognizer.write("face_recognizer.yml") # save the trained model to a file, so we can use it later

#print what lives inside the face_result[0]
print(f'looking inside the face result: {face_result[0]}')
print(f'flooking inside the boxes:{face_result[0].boxes}')
print(f'flooking inside the boxes coordinates xyxy  :{face_result[0].boxes.xyxy}')




#test code to verify images are being read correctly
#Image.fromarray(photo[:,:,::-1]).show()

#Image.fromarray(processed_image[:,:,::-1]).show()
#Image.fromarray(face[:,:,::-1]).show()
Image.fromarray(faces[0]).show()

# catch it on the the screen
cap=cv2.VideoCapture(0) ## creates a video capture object to capture the video from the webcam
#add a list of names, rather than just numbers, so we can see the names of the people being recognized
names={0:'Mariya',1:'Mario'} # if we had more than one person, we would add them here with a different number for each person
max_distance=70 # this is the maximum distance for a match to be considered valid, if the distance is greater than this, it will be considered a non-match
#read the model that is already familiar with the faces we trained it on
face_recognizer.read("face_recognizer.yml") # load the trained model from the file.


while cv2.waitKey(1) != ord("x"):
   
    
    ret, frame=cap.read()
    face_result=face_model(frame, verbose=False)# 
    processed_feed=face_result[0].plot()
    for box in face_result[0].boxes.xyxy:
        left, top, right, bottom=box.int()
        #we use 0 for the one and only face in the photo
        # # we use into , as there are no decimal points in pixels, and we need to convert the float values to integers
        face=frame[top:bottom, left:right]
        # turn into a gray color for the face recognizer to work with and reassign the face variable to the new gray image
        face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        #resize the face to a standard size, so the recognizer can work with it
        face=cv2.resize(face, (200,200))
        pred_label, distance=face_recognizer.predict(face)
        if distance>max_distance:

            name="unknown"
        else:
            name=names[pred_label]
        cv2.putText(
        processed_feed,
        name+"|| "+str(int(distance)),
        (int(left),int(bottom+20)),
        0,
        0.8,
        (255,255,255),
        2
        )
    cv2.imshow("my window", processed_feed)
                
# Release the hardware back to the operating system    
cap.release()
# Close all OpenCV windows to clear the screen
cv2.destroyAllWindows()