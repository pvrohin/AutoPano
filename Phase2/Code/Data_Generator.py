import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random

def main():
    #Read an image from Phase2Data/Train
    image = cv2.imread('Phase2/Data/Train/1.jpg')

    #Print and get the size of the image
    print(image.shape)

    #Get the size of the patch
    MP, NP = 100, 100

    #Obtain a random patch (PA of size MP×NP) from the image (IA) of size M×N) with M>MP and N>NP
    #All the pixels in the patch will lie within the image after warping the random extracted patch. Think about where you have to extract the patch PA from IA if maximum possible perturbation is in [−ρ,ρ]
    ρ = 10
    patch = image[random.randint(0, image.shape[0]-MP), random.randint(0, image.shape[1]-NP)]

    #Perform a random perturbation in the range [−ρ,ρ] to the corner points (top left corner, top right corner, left bottom corner and right bottom corner – not the corners in computer vision sense) of PA in IA. You might also have to add a random translation amount (fixed for all 4 points) such that your network would work for translated images as well. The random perturbation (without translation) is illustrated in the figures below.
    perturbation = random.randint(-ρ, ρ)
    translation = random.randint(-ρ, ρ)
    rotation = random.randint(-ρ, ρ)
    scale = random.randint(-ρ, ρ)
    shear = random.randint(-ρ, ρ)
    
    #Use cv2.getPerspectiveTransform and np.linalg.inv to implement this part.
    H = cv2.getPerspectiveTransform(patch, image)
    H_inv = np.linalg.inv(H)
    
    #Use cv2.warpPerspective to implement this part.
    warped_patch = cv2.warpPerspective(patch, H_inv, image.shape)
    
    #Use the value of H_inv to warp IA and obtain IB. Use cv2.warpPerspective to implement this part. Now, we can extract the patch PB using the corners in CA (work the math out and convince yourself why this is true). This is shown in the figure below.
    warped_image = cv2.warpPerspective(image, H_inv, image.shape)
    
    #Extract the patch PB using the corners in CA (work the math out and convince yourself why this is true). This is shown in the figure below.
    patch_b = warped_image[random.randint(0, warped_image.shape[0]-MP), random.randint(0, warped_image.shape[1]-NP)]

    #Print the size of the patch
    print(patch_b.shape)

    #Print the size of the warped patch
    print(warped_patch.shape)

    #Print the size of the warped image
    print(warped_image.shape)

if __name__ == "__main__":
    main()