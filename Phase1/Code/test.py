import numpy as np
import cv2
import os
from skimage.feature import peak_local_max
from scipy.ndimage import maximum_filter

def detectCorners(imgs, choice):
    images = imgs.copy()
    print("detecting corners ...")
    detected_corners = []
    cmaps = []
    corner_images = []
    for i in images:
        image = i.copy()
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray_image = np.float32(gray_image)


        if(choice == 1):
            print("using Harris corner detection method.")
            corner_strength = cv2.cornerHarris(gray_image,2,3,0.001)
            corner_strength[corner_strength<0.01*corner_strength.max()] = 0
            detected_corner = np.where(corner_strength>0.0001*corner_strength.max())
            detected_corners.append(detected_corner)
            cmaps.append(corner_strength)
            image[corner_strength > 0.0001*corner_strength.max()]=[0,0,255]
            corner_images.append(image)
        else:
            print("using Shi-Tomashi corner detection method.")
            dst = cv2.goodFeaturesToTrack(gray_image, 1000 ,0.01, 10)
            dst = np.int0(dst)
            detected_corners.append(dst)
            for c in dst:
                x,y = c.ravel()
                cv2.circle(image,(x,y),3,(0, 0, 255),-1) 
                          
            corner_images.append(image)
            cmap = np.zeros(gray_image.shape) #not sure what to do
            cmaps.append(cmap)
    #filter detected corners
    #remove the corner one
    return detected_corners, cmaps, corner_images

def main():
    img = cv2.imread("/Users/rohin/Documents/Computer Vision/AutoPano/Phase1/Data/Train/Set1/1.jpg")
    imgs = [img]
    choice = 1
    detected_corners, cmaps, corner_images = detectCorners(imgs, choice)
    print("Detected Corners: ", detected_corners)
    print("Corner Maps: ", cmaps)
    #Display corner image
    cv2.imshow("Corner Image", corner_images[0])
    cv2.waitKey(0)

if __name__ == "__main__":
    main()