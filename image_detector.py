# import the necessary libraries
import cv2
from ultralytics import YOLO


##create the model using the yolov8n.pt weights
model=YOLO("yolov8n.pt")
face_model=YOLO("yolov8m-face.pt")
# create an instance of the photo 

photo=cv2.imread("Gemini_Generated_Image5_.png") ## reads the photo path
## Add the photo to the model for detection
result=model(photo)
face_result=face_model(photo)
object_detection=result[0].plot()
face=face_result[0].plot(img=object_detection)
## detect objects


print(result)    

cv2.imshow("my window", face)## shows the photo and assigns the name
## of the window and provides the photo to be shown

## assigns the position of the window on the screen
cv2.moveWindow("my window", 100, 100)


# provides wait time for the window to be displayed before it is closed 
cv2.waitKey(500000)

## Destroy all the windows created by OpenCV
cv2.destroyAllWindows()
