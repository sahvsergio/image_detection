import cv2
from ultralytics import YOLO


## Create a vide capture
cap= cv2.VideoCapture(0) ## 0 is the default camera, if you have multiple cameras, 
#you can change the index to 1, 2, etc.



## add models for object detection and face detection
model=YOLO("yolov8n.pt")
face_model=YOLO("yolov8m-face.pt")
# create a while loop to continuously read frames from the camera
while cv2.waitKey(1)!= ord("x"):
    _,frame=cap.read() ## read the frames from the camera
    cv2.imshow("my window", frame)## shows the photo and assigns the name
    ## of the window and provides the photo to be shown
    # ## assigns the position of the window on the screen
    cv2.moveWindow("my window", 100, 100)


# provides wait time for the window to be displayed before it is closed 
cv2.waitKey(500000)
cap.release() ## release the camera
## Destroy all the windows created by OpenCV
cv2.destroyAllWindows()
