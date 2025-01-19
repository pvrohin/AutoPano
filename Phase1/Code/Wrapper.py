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

# Add any python libraries here

# Corner Detection using Harris Corner Detection
# def CornerDetection(img):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
#     # Detect corners using Harris Corner Detection
#     corners = cv2.cornerHarris(gray, 2, 3, 0.001)
     
#     # Dilate corner image to enhance corner points
#     corners = cv2.dilate(corners, None)
    
#     # Create a copy of the original image to display corners
#     img_with_corners = img.copy()
    
#     # Display the corners on the image
#     img_with_corners[corners > 0.01 * corners.max()] = [0, 0, 255]
    
#     cv2.imshow('corners', img_with_corners)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
     
#     # Return the corners
#     return corners

def extract_corners(image, use_harris=True, block_size=2, ksize=3, k=0.001, num_corners=100):
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply either Harris corner detection or Shi-Tomasi corner detection
    if use_harris:
        # Perform Harris corner detection
        corners = cv2.cornerHarris(gray, block_size, ksize, k)
    
        # Refine the corners by applying a threshold
        corners[corners < 0.001 * corners.max()] = 0
        #corners = cv2.threshold(corners, 0.001 * corners.max(), 255, cv2.THRESH_BINARY)[1]
        # Find the coordinates of the corners
        corner_coords = np.argwhere(corners > 0.01 * corners.max())
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
    
    print(corner_coords1)
    print(corners1)
    
    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

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
