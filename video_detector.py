# import the necessary libraries
import os 
import cv2
import torch
from ultralytics import YOLO

num_threads=os.cpu_count()
# set the number of threads for PyTorch to use  
torch.set_num_threads(num_threads)
print("Running on CPU with {} threads".format(num_threads))
# 4. Execute inference on CPU utilizing all 8 threads


##create the model using the yolov8n.pt weights
model=YOLO("yolov8n.pt")
face_model=YOLO("yolov8m-face.pt")
cap=cv2.VideoCapture(0) ## creates a video capture object to capture the video from the webcam

while cv2.waitKey(1) != ord("x"):
    # read each frame from the webcam
    _,frame=cap.read()
    result=model(frame)
    face_result=face_model(frame)
    object_detection=result[0].plot()
    face=face_result[0].plot(img=object_detection)
    results = model.predict(source=face, device="cpu", verbose=False)
    annotated_frame=results[0].plot()
    cv2.imshow('my window',results[0].plot()) ## shows the photo and assigns the name of the window and provides the photo to be shown
    cv2.moveWindow("my window",100,100)



cv2.waitKey(5000)
cap.release()
cv2.destroyAllWindows()


