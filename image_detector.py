# import the necessary libraries
import cv2

# create an instance of the photo 

photo=cv2.imread("3586.jpg") ## reads the photo path
cv2.imshow("my window", photo)## shows the photo and assigns the name
## of the window and provides the photo to be shown

## assigns the position of the window on the screen
cv2.moveWindow("my window", 100, 100)


# provides wait time for the window to be displayed before it is closed 
cv2.waitKey(5000)

## Destroy all the windows created by OpenCV
cv2.destroyAllWindows()