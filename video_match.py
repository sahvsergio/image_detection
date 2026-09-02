
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
