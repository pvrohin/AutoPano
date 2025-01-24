#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import numpy as np
import cv2
import os
from skimage.feature import peak_local_max
from scipy.ndimage import maximum_filter

# Add any python libraries here

def extract_corners(image, use_harris=True, block_size=2, ksize=3, k=0.001, num_corners=1000):
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply either Harris corner detection or Shi-Tomasi corner detection
    if use_harris:
        # Perform Harris corner detection
        corners = cv2.cornerHarris(gray, block_size, ksize, k)
    
        # Refine the corners by applying a threshold
        corners[corners < 0.001 * corners.max()] = 0
    
        # Find the coordinates of the corners
        corner_coords = np.argwhere(corners > 0.01 * corners.max())
        
		# Convert the coordinates to a list of (x, y) tuples
        corner_coords = [corner[::-1] for corner in corner_coords]  # Reverse x-y order

    else:
        # Perform Shi-Tomasi corner detection
        corners = cv2.goodFeaturesToTrack(gray, num_corners, 0.01, 10)
        corner_coords = np.int0(corners).reshape(-1, 2).tolist()

    # Draw the detected corners on the image
    image_with_corners = image.copy()

    if use_harris:
        # Harris corner detection produces a corner score image
        image_with_corners[corners > 0.01 * corners.max()] = [0, 0, 255]
    else:
        # Shi-Tomasi corner detection returns corner points directly
        for corner in corner_coords:
            x, y = corner
            cv2.circle(image_with_corners, (x, y), 3, (0, 0, 255), -1)

    return corners, corner_coords, image_with_corners

# Perform Adaptive Non-Maximal Suppression (ANMS) to select the best corners from the detected corners
def ANMS(image, corner_map, N_Best=500):

    local_maxima = peak_local_max(corner_map, 15)

    N_Strong = len(local_maxima)

    r = [np.Inf for i in range(N_Strong)]

    ED = 0

    count = 0

    for i in range(N_Strong):
        for j in range(N_Strong):
            if corner_map[local_maxima[i][0]][local_maxima[i][1]] < corner_map[local_maxima[j][0]][local_maxima[j][1]]:
                ED = np.sqrt((local_maxima[i][0] - local_maxima[j][0])**2 + (local_maxima[i][1] - local_maxima[j][1])**2)
            if ED < r[i]:
                r[i] = ED
                count += 1

    if count < N_Best:
        N_Best = count

    #Sort the r list in descending order and get the N_Best corners without using zip function
    N_Best_Corners = [local_maxima[i] for i in np.argsort(r)[::-1][:N_Best]]
  
    #print(N_Best_Corners)
    
    #Show the N_Best corners on the image
    for i in range(len(N_Best_Corners)):
       cv2.circle(image, (int(N_Best_Corners[i][1]), int(N_Best_Corners[i][0])), 3, (0, 0, 255), -1)
    
    # cv2.imshow("N_Best_Corners", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("N_Best_Corners.jpg", image)  

    return N_Best_Corners

def main(): 
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """
    # Read images from Data/Set1 as a preliminary implementation
    # You can change the path to any other dataset
    img1 = cv2.imread('/Users/rohin/Documents/Computer Vision/AutoPano/Phase1/Data/Train/Set1/1.jpg')
    img2 = cv2.imread('/Users/rohin/Documents/Computer Vision/AutoPano/Phase1/Data/Train/Set1/2.jpg')
    img3 = cv2.imread('/Users/rohin/Documents/Computer Vision/AutoPano/Phase1/Data/Train/Set1/3.jpg')

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""
    # Perform corner detection on the images
    corners1, corner_coords1, image_with_corners1 = extract_corners(img1)
    corners2, corner_coords2, image_with_corners2 = extract_corners(img2)
    corners3, corner_coords3, image_with_corners3 = extract_corners(img3)
    
	#Create an output folder if it does not exist
    if not os.path.exists('Output'):
        os.makedirs('Output')
    
    # Create an output folder and save the images with detected corners
    cv2.imwrite('Output/corners1.png', image_with_corners1)
    cv2.imwrite('Output/corners2.png', image_with_corners2)
    cv2.imwrite('Output/corners3.png', image_with_corners3) 
    
    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""
    
	# Perform ANMS on the corner maps
    N_Best_Corners1 = ANMS(image_with_corners1, corners1)
    N_Best_Corners2 = ANMS(image_with_corners2, corners2)
    N_Best_Corners3 = ANMS(image_with_corners3, corners3)
    
	#Store the ANMS output in the Output folder
    cv2.imwrite('Output/anms1.png', image_with_corners1)
    cv2.imwrite('Output/anms2.png', image_with_corners2)
    cv2.imwrite('Output/anms3.png', image_with_corners3)

    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""

    """
	Refine: RANSAC, Estimate Homography
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
