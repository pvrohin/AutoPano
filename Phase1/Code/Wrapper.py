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
import argparse
import os
from skimage.feature import peak_local_max
from scipy.linalg import svd

def corner_detection(images, use_harris=True, block_size=2, ksize=3, k=0.04, num_corners=1000):
	
    corner_score_images = []
    corners = []
    corner_marked_images = []
	
    if use_harris:
        for image in images:
            image = image.copy()

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			
            gray = np.float32(gray)

            harris_corners = cv2.cornerHarris(gray, block_size, ksize, k)
			
            harris_corners = cv2.dilate(harris_corners, None)
			
            image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
			
            corner_marked_images.append(image)
			
            corners.append(np.transpose(np.where(harris_corners > 0.01 * harris_corners.max())))
			
            harris_corners[harris_corners < 0.01 * harris_corners.max()] = 0
			
            corner_score_images.append(harris_corners)
			
    else:
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			
            #Follow the same format as the Harris corner detection
            shi_tomasi_corners = cv2.goodFeaturesToTrack(gray, num_corners, 0.01, 10)
			
            #Mark the detected corners on the image
            image_with_corners = image.copy()
            for corner in shi_tomasi_corners:
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(image_with_corners, (x, y), 3, (0, 0, 255), -1)
				
            corner_marked_images.append(image_with_corners)
			
            corners.append(np.array([corner[0] for corner in shi_tomasi_corners]))

            corner_score_images.append(shi_tomasi_corners)
			
    return corner_score_images, corners, corner_marked_images
	
def anms(images,corner_score_images,N_best=400):
	anms_images=[]
	corners=[]
	for i in range(len(images)):
		image=images[i].copy()
		corner_score_image=corner_score_images[i]
		local_maximas=peak_local_max(corner_score_image, min_distance=10)
		N_strong=len(local_maximas)
		if N_strong == 0:
			print(f"No corners found in image {i}")
			continue
			
		r=np.ones(N_strong)*np.inf
		x=np.zeros(N_strong)
		y=np.zeros(N_strong)
		for i in range(N_strong):
			for j in range(N_strong):
				if corner_score_image[local_maximas[i][0]][local_maximas[i][1]]<corner_score_image[local_maximas[j][0]][local_maximas[j][1]]:
					distance=np.sqrt((local_maximas[i][0]-local_maximas[j][0])**2+(local_maximas[i][1]-local_maximas[j][1])**2)
					if distance<r[i]:
						r[i]=distance
						x[i]=local_maximas[i][1]  # Note: swapped x and y to match OpenCV's coordinate system
						y[i]=local_maximas[i][0]
		
		# Sort corners by radius and take top N_best
		sorted_indices = np.argsort(r)[::-1]
		N_final = min(N_best, N_strong)
		x = x[sorted_indices[:N_final]]
		y = y[sorted_indices[:N_final]]
		
		anms_image=image.copy()
		for i in range(N_final):
			cv2.circle(anms_image,(int(x[i]),int(y[i])),3,(0,0,255),-1)
		anms_images.append(anms_image)
		# Return corners as a list of [x,y] pairs
		corners.append(np.column_stack((x, y)))
	return anms_images,corners

def feature_descriptors(images,corners):
	feature_vectors=[]
	corresponding_corners=[]
	for i in range(len(images)):
		image=images[i].copy()
		gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		corners_in_image=corners[i]
		feature_vectors_image=[]
		corresponding_corners_image=[]
		for j in range(len(corners_in_image)):
			# Convert corner coordinates to integers
			x = int(corners_in_image[j][0])
			y = int(corners_in_image[j][1])
			if x-20<0 or x+20>gray.shape[1] or y-20<0 or y+20>gray.shape[0]:
				continue
			patch=gray[y-20:y+20,x-20:x+20]
			patch=cv2.GaussianBlur(patch,(5,5),0)
			patch=cv2.resize(patch,(8,8))
			patch=patch.flatten()
			patch=(patch-np.mean(patch))/np.std(patch)
			feature_vectors_image.append(patch)
			corresponding_corners_image.append([x,y])
		corresponding_corners.append(corresponding_corners_image)
		feature_vectors.append(feature_vectors_image)
	return feature_vectors, corresponding_corners
			
	
def feature_matching(image1, image2,feature_vectors1, feature_vectors2,corresponding_corners1, corresponding_corners2):
	matched_corners = []
	matched_indexes = []
	# For each feature vector in image 1, find best and second best matches in image 2
	for i in range(len(feature_vectors1)):
		distances = []
		for j in range(len(feature_vectors2)):
			# Calculate Euclidean distance between feature vectors
			distance = np.linalg.norm(feature_vectors1[i] - feature_vectors2[j])
			distances.append(distance)
		
		distances = np.array(distances)
		sorted_indices = np.argsort(distances)
		
		# Apply ratio test - compare best match to second best match
		ratio = distances[sorted_indices[0]] / distances[sorted_indices[1]]
		if ratio < 0.8: # Threshold of 0.8 for ratio test
			matched_corners.append([corresponding_corners1[i], corresponding_corners2[sorted_indices[0]]])
			matched_indexes.append([i, sorted_indices[0]])
	
	# Check if enough matches were found
	if len(matched_corners) < 25:
		print('Not enough matches found (minimum 25 required)')
		return None, None
		
	# Convert matches to format needed for visualization
	keypoints1 = [cv2.KeyPoint(float(corner[0]), float(corner[1]), 1) for corner in corresponding_corners1]
	keypoints2 = [cv2.KeyPoint(float(corner[0]), float(corner[1]), 1) for corner in corresponding_corners2]
	
	matches = [cv2.DMatch(pair[0], pair[1], 0) for pair in matched_indexes]
	
	# Draw the matches
	matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)
	
	return np.array(matched_corners), matched_image

def find_homography(src_points, dest_points):
    A = []
    for src, dest in zip(src_points, dest_points):
        x, y = src
        xp, yp = dest
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    A = np.array(A)
    _, _, V = svd(A)
    H = V[-1, :].reshape((3, 3))
    return H

def ransac(image1,image2,matched_corners):
    corners1=matched_corners[:,0]
    corners2=matched_corners[:,1]
    Nmax=4000
    max_inliers=0
    best_H=None
    thresh=10
    for i in range(Nmax):
        indices=np.random.choice(len(matched_corners),4)
        corners1_sample=corners1[indices]
        corners2_sample=corners2[indices]
        H=find_homography(corners1_sample,corners2_sample)
        corners1_homogeneous=np.vstack((corners1.T,np.ones(len(corners1))))
        try:
            corners1_transformed=np.dot(H,corners1_homogeneous)
        except:
            continue
        corners1_transformed=corners1_transformed[:2,:]/(corners1_transformed[2,:]+1e-10)
        
        sum_squared_distance=np.sum((corners1_transformed.T-corners2)**2,axis=1)
        
        inliers=np.sum(sum_squared_distance<thresh)
        
        if inliers>max_inliers:
            max_inliers=inliers
            best_H=H
            matched_corners_inliers=matched_corners[sum_squared_distance<thresh]
            matched_corners_inliers_index=np.where(sum_squared_distance<thresh)
    keypoints1=[]
    keypoints2=[]
    for i in corners1:
        keypoints1.append(cv2.KeyPoint(float(i[0]),float(i[1]),1))
    for i in corners2:
        keypoints2.append(cv2.KeyPoint(float(i[0]),float(i[1]),1))

    matches=[]
    for i in matched_corners_inliers_index[0]:
        matches.append(cv2.DMatch(i,i,0))
    
    out_image=cv2.drawMatches(image1,keypoints1,image2,keypoints2,matches,None)
    return best_H,matched_corners_inliers,out_image

def warp_blend(image1,image2,H):
	image1_boundaries=np.array([[0.0,0.0],[0.0,image1.shape[0]],[image1.shape[1],0.0],[image1.shape[1],image1.shape[0]]]).reshape(-1,1,2)
	image1_boundaries_transformed=cv2.perspectiveTransform(image1_boundaries,H)

	image2_boundaries=np.array([[0.0,0.0],[0.0,image2.shape[0]],[image2.shape[1],0.0],[image2.shape[1],image2.shape[0]]]).reshape(-1,1,2)

	final_boundaries=np.concatenate((image1_boundaries_transformed,image2_boundaries),axis=0)
	
	[x_min,y_min]=np.intp(np.min(final_boundaries,axis=0)[0])
	[x_max,y_max]=np.intp(np.max(final_boundaries,axis=0)[0])

	translation_matrix=np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]])

	image1_transformed=cv2.warpPerspective(image1.copy(),translation_matrix.dot(H),(x_max-x_min,y_max-y_min))
	
	final_image=image1_transformed.copy()
	final_image[-y_min:image2.shape[0]-y_min,-x_min:image2.shape[1]-x_min]=image2.copy()

	indices=np.where(image2==[0,0,0])
	y=indices[0]-y_min
	x=indices[1]-x_min

	final_image[y,x]=image1_transformed[y,x]
	return final_image

def main():
    
	# Add any Command Line arguments here
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--ImagePath', default='/Users/rohin/Documents/Computer Vision/AutoPano/Phase1/Data/Train/Set1', help='Path to image directory')
	Args = Parser.parse_args()
	images_path = Args.ImagePath

	"""
	Read a set of images for Panorama stitching
	"""
	images = []
	dataset_name = images_path.split('Data/')[-1]
	results_path = './Results/' + dataset_name + '/'
	if not os.path.exists(results_path):
		os.makedirs(results_path)
		
	images_names = sorted([name for name in os.listdir(images_path) if name.endswith('.jpg')])
	num_images = len(images_names)
	for i in range(num_images):
		image_path = os.path.join(images_path, images_names[i])
		image = cv2.imread(image_path)
		if image is None:
			print(f"Error: Could not read image {image_path}")
			continue
		images.append(image)
		
	if not images:
		print("No images were successfully loaded. Exiting.")
		return
		
	image = images[0].copy()
	corner_images_i = 0
	anms_images_i = 0
	feature_matching_i = 0
	ransac_i = 0
	pano_i = 0
	
	for i in range(1, num_images):
		current_images = [image, images[i]]
		"""
		Corner Detection
		Save Corner detection output as corners.png
		"""
		corner_score_images, corners, corner_marked_images = corner_detection(current_images)
		for j in range(len(corner_marked_images)):
			cv2.imwrite(results_path + 'corner_images_' + str(corner_images_i) + '.png', corner_marked_images[j])
			corner_images_i += 1
		"""
		Perform ANMS: Adaptive Non-Maximal Suppression
		"""
		anms_images,corners=anms(current_images,corner_score_images)
		for j in range(len(anms_images)):
			cv2.imwrite(results_path + 'anms_images_' + str(anms_images_i) + '.png', anms_images[j])
			anms_images_i += 1
		"""
		Feature Descriptors
		"""
		feature_vectors, corresponding_corners = feature_descriptors(current_images, corners)
		sample_feature_vectors = feature_vectors[0][0]
		sample_feature_vectors = sample_feature_vectors.reshape((8,8))*255
		cv2.imwrite(results_path + 'feature_descriptor_image.png', sample_feature_vectors)
		"""
		Feature Matching
		Save Feature Matching output as matching.png
		"""
		matched_corners,feature_matched_image=feature_matching(current_images[0],current_images[1],feature_vectors[0],feature_vectors[1],corresponding_corners[0],corresponding_corners[1])
		if matched_corners is None:
			print("Feature matching failed - not enough matches found")
			continue
		print(f"Found {len(matched_corners)} matched corners")
		cv2.imwrite(results_path+'feature_matched_image_'+str(feature_matching_i)+'.png',feature_matched_image)
		feature_matching_i+=1
		
		"""
		RANSAC
		"""
		ransac_H,matched_corners_inliers,ransac_image=ransac(current_images[0],current_images[1],matched_corners)
		cv2.imwrite(results_path+'ransac_image_'+str(ransac_i)+'.png',ransac_image)
		ransac_i+=1
		if ransac_H is None:
			print("RANSAC failed - could not find a good homography")
			continue
		print("Successfully found homography matrix")
		
		"""
		Warp and Blend
		"""
		image=warp_blend(current_images[0],current_images[1],ransac_H)
		if image is None:
			print("Warp and blend failed")
			continue
		print("Successfully created panorama")
		cv2.imwrite(results_path+'pano_image_'+str(pano_i)+'.png',image)
		pano_i+=1
		
if __name__ == "__main__":
    main()