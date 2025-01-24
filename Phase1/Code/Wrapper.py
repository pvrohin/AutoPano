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
def ANMS(image, corner_map, N_Best=500, min_distance=15, radius_threshold=1e-3):
    """
    Adaptive Non-Maximal Suppression (ANMS) for corner detection.
    
    Parameters:
        image (numpy.ndarray): The input image on which corners are drawn.
        corner_map (numpy.ndarray): The corner strength map.
        N_Best (int): The maximum number of corners to retain after suppression.
        min_distance (int): Minimum distance between local maxima.
        radius_threshold (float): Threshold for the suppression radius.

    Returns:
        list: N_Best corners as (x, y) coordinates.
        numpy.ndarray: Image with the corners drawn.
    """
    # Normalize the corner map to ensure consistent scaling
    corner_map = corner_map / corner_map.max()

    # Apply a threshold to the corner map to remove weak corners
    corner_map[corner_map < 0.01] = 0

    # Find local maxima in the corner map
    local_maxima = peak_local_max(corner_map, min_distance=min_distance)
    N_Strong = len(local_maxima)

    # Initialize suppression radius (r) with infinity
    r = [np.Inf for _ in range(N_Strong)]

    # Calculate suppression radius for each local maxima
    for i in range(N_Strong):
        for j in range(N_Strong):
            if corner_map[local_maxima[j][0], local_maxima[j][1]] > corner_map[local_maxima[i][0], local_maxima[i][1]]:
                eu_dist = (local_maxima[j][0] - local_maxima[i][0])**2 + (local_maxima[j][1] - local_maxima[i][1])**2
                r[i] = min(r[i], eu_dist)

    # Filter corners based on suppression radius threshold
    valid_indices = [i for i, radius in enumerate(r) if radius > radius_threshold]
    sorted_indices = np.argsort(r)[::-1]  # Sort indices by descending radius

    # Select top N_Best corners
    selected_indices = [i for i in sorted_indices if i in valid_indices][:N_Best]
    N_Best_Corners = [local_maxima[i] for i in selected_indices]

    # Draw the selected corners on the image
    output_image = image.copy()
    for corner in N_Best_Corners:
        cv2.circle(output_image, (int(corner[1]), int(corner[0])), 3, (0, 255, 0), -1)

    return N_Best_Corners, output_image


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
    anms_corners1, image_with_corners1 = ANMS(img1, corners1)
    anms_corners2, image_with_corners2 = ANMS(img2, corners2)
    anms_corners3, image_with_corners3 = ANMS(img3, corners3)
    
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
