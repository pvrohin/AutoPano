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

# Extract feature descriptors from the detected corners
def get_feature_descriptor(img, x, y):
    # 1) Extract a 41×41 patch centered around (x, y)
    size = 41
    half = size // 2
    patch = img[y-half:y+half+1, x-half:x+half+1]

    # Handle boundary cases
    if patch.shape[0] != size or patch.shape[1] != size:
        return None  # or pad if needed

    # 2) Apply Gaussian blur
    blurred = cv2.GaussianBlur(patch, (5, 5), 0)

    # 3) Sub-sample to 8×8
    sub = cv2.resize(blurred, (8, 8), interpolation=cv2.INTER_AREA)

    # 4) Reshape to 64×1
    descriptor = sub.reshape(64).astype(np.float32)

    # 5) Standardize (zero mean, unit variance)
    mean, std = cv2.meanStdDev(descriptor)
    descriptor = (descriptor - mean[0][0]) / (std[0][0] + 1e-6)

    return descriptor

def extract_feature_descriptors(image, corners):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptors = []
    for corner in corners:
        x, y = int(corner[1]), int(corner[0])
        descriptor = get_feature_descriptor(gray, x, y)
        if descriptor is not None:
            descriptors.append(descriptor)
    return np.array(descriptors)

def match_features(desc1, desc2, ratio_threshold=0.75):
    matches = []
    for i, d1 in enumerate(desc1):
        distances = np.linalg.norm(desc2 - d1, axis=1)
        if len(distances) < 2:
            continue
        sorted_indices = np.argsort(distances)
        best_match = sorted_indices[0]
        second_best_match = sorted_indices[1]
        if distances[best_match] < ratio_threshold * distances[second_best_match]:
            matches.append((i, best_match))
    return matches

def ransac_homography(matches, keypoints1, keypoints2, threshold=5.0, max_iterations=1000):
    max_inliers = []
    best_H = None

    for _ in range(max_iterations):
        # Randomly select 4 matches
        sample_indices = np.random.choice(len(matches), 4, replace=False)
        pts1 = np.float32([keypoints1[matches[i][0]].pt for i in sample_indices])
        pts2 = np.float32([keypoints2[matches[i][1]].pt for i in sample_indices])

        # Compute homography
        H, _ = cv2.findHomography(pts1, pts2, 0)

        # Compute inliers
        inliers = []
        for i, (m1, m2) in enumerate(matches):
            pt1 = np.float32([keypoints1[m1].pt])
            pt2 = np.float32([keypoints2[m2].pt])
            projected_pt2 = cv2.perspectiveTransform(np.array([pt1]), H)[0][0]
            ssd = np.sum((pt2 - projected_pt2) ** 2)
            if ssd < threshold:
                inliers.append((m1, m2))

        # Update best homography if more inliers are found
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            best_H = H

        # Early exit if a sufficient number of inliers is found
        if len(max_inliers) > 0.9 * len(matches):
            break

    # Recompute homography using all inliers
    if best_H is not None:
        pts1 = np.float32([keypoints1[m[0]].pt for m in max_inliers])
        pts2 = np.float32([keypoints2[m[1]].pt for m in max_inliers])
        best_H, _ = cv2.findHomography(pts1, pts2, 0)

    return best_H, max_inliers

def blend_images(img1, img2, H):
    # Warp img2 to img1's plane
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # Get the canvas dimesions
    corners_img2 = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)
    warped_corners_img2 = cv2.perspectiveTransform(corners_img2, H)
    all_corners = np.concatenate((corners_img2, warped_corners_img2), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:height1 + translation_dist[1], translation_dist[0]:width1 + translation_dist[0]] = img1

    return output_img

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
    
    # Print size of image 1
    print(img1.shape)


    print(anms_corners1)
    
	#Store the ANMS output in the Output folder
    cv2.imwrite('Output/anms1.png', image_with_corners1)
    cv2.imwrite('Output/anms2.png', image_with_corners2)
    cv2.imwrite('Output/anms3.png', image_with_corners3)

    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""
    # Extract feature descriptors for the detected corners
    descriptors1 = extract_feature_descriptors(img1, anms_corners1)
    descriptors2 = extract_feature_descriptors(img2, anms_corners2)
    descriptors3 = extract_feature_descriptors(img3, anms_corners3)

    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""
    matches = match_features(descriptors1, descriptors2)

    keypoints1 = [cv2.KeyPoint(float(c[1]), float(c[0]), 1) for c in anms_corners1]
    keypoints2 = [cv2.KeyPoint(float(c[1]), float(c[0]), 1) for c in anms_corners2]

    good_matches = [cv2.DMatch(m[0], m[1], 0) for m in matches]

    matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None)

    cv2.imshow('Matches', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('Output/matching.png', matched_image)

    """
	Refine: RANSAC, Estimate Homography
	"""
    # RANSAC for homography
    H, inliers = ransac_homography(matches, keypoints1, keypoints2)

    print("Homography Matrix:\n", H)
    print("Number of inliers:", len(inliers))

    # Visualize inliers
    inlier_matches = [cv2.DMatch(m[0], m[1], 0) for m in inliers]
    inlier_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, inlier_matches, None)

    cv2.imshow('Inliers', inlier_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('Output/inliers.png', inlier_image)

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
    # Blend images to create panorama
    panorama = blend_images(img1, img2, H)

    cv2.imshow('Panorama', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('Output/mypano.png', panorama)


if __name__ == "__main__":
    main()
