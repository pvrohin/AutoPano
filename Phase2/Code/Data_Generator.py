#Load an image from Data folder
import cv2
import os

image_path = os.path.join('../Data/Train/1.jpg')
image = cv2.imread(image_path)
print(image.shape)



