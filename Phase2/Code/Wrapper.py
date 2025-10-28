#!/usr/bin/env python

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
import argparse
import os
import sys
import torch
import glob
from Network.Network_Unsupervised import HomographyModel
from kornia.geometry.transform import HomographyWarper

# Don't generate pyc codes
sys.dont_write_bytecode = True


def predict_homography(model, patch1, patch2):
    """
    Predict homography using the trained deep learning model
    Input: patch1, patch2 - image patches
    Output: H4pt (8 values representing homography 4-point displacement)
    """
    # Convert to grayscale if needed
    if len(patch1.shape) == 3:
        patch1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
    if len(patch2.shape) == 3:
        patch2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY)
    
    # Resize to 128x128
    patch1 = cv2.resize(patch1, (128, 128))
    patch2 = cv2.resize(patch2, (128, 128))
    
    # Convert to torch tensors and normalize
    patch1_tensor = torch.from_numpy(patch1).float().unsqueeze(0).unsqueeze(0) / 255.0
    patch2_tensor = torch.from_numpy(patch2).float().unsqueeze(0).unsqueeze(0) / 255.0
    
    # Stack patches
    combined = torch.cat([patch1_tensor, patch2_tensor], dim=1)  # (1, 2, 128, 128)
    
    # Predict H4pt
    with torch.no_grad():
        H4pt = model(combined)  # (1, 8)
        H4pt = H4pt.squeeze(0).cpu().numpy()  # (8,)
    
    return H4pt


def h4pt_to_homography(h4pt, corners):
    """
    Convert H4pt (4-point displacement) to full homography matrix using DLT
    Input: h4pt - 8 values, corners - corner coordinates of patch1
    Output: H - 3x3 homography matrix
    """
    # Reshape H4pt to (4, 2)
    H4pt_2d = h4pt.reshape(4, 2)
    
    # Original corners (assuming patch is from full image)
    C_A = corners  # (4, 2)
    
    # Perturbed corners
    C_B = C_A + H4pt_2d
    
    # Use DLT to compute homography
    H = compute_homography_dlt(C_A, C_B)
    
    return H


def compute_homography_dlt(pts1, pts2):
    """
    Direct Linear Transform (DLT) to compute homography from point correspondences
    Input: pts1, pts2 - arrays of corresponding points (N, 2)
    Output: H - 3x3 homography matrix
    """
    # Construct the A matrix
    A = []
    for i in range(len(pts1)):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        
        A.append([-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
    
    A = np.array(A)
    
    # SVD to find null space
    _, _, V = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    
    return H


def find_best_patches(img1, img2):
    """
    Find best patch pairs from two images for homography estimation
    Uses corner detection and matching to find good patches
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    
    # Detect corners
    corners1 = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corners2 = cv2.goodFeaturesToTrack(gray2, maxCorners=100, qualityLevel=0.01, minDistance=10)
    
    if corners1 is None or corners2 is None:
        # Fallback: use image centers
        h1, w1 = gray1.shape
        h2, w2 = gray2.shape
        
        # Take patches from center of images
        patch1 = gray1[max(0, h1//2-64):h1//2+64, max(0, w1//2-64):w1//2+64]
        patch2 = gray2[max(0, h2//2-64):h2//2+64, max(0, w2//2-64):w2//2+64]
        
        # Define corner coordinates
        corners_coord = np.array([
            [max(0, w1//2-64), max(0, h1//2-64)],      # top-left
            [w1//2+64, max(0, h1//2-64)],              # top-right
            [max(0, w1//2-64), h1//2+64],              # bottom-left
            [w1//2+64, h1//2+64]                       # bottom-right
        ])
        
        return patch1, patch2, corners_coord
    
    # Extract multiple patches for robust estimation
    num_patches = 5
    best_patches = []
    
    for _ in range(num_patches):
        # Random corner from image1
        idx1 = np.random.randint(len(corners1))
        x1, y1 = corners1[idx1].ravel()
        
        # Get patch around this corner
        y_start = int(max(0, y1 - 64))
        y_end = int(min(gray1.shape[0], y1 + 64))
        x_start = int(max(0, x1 - 64))
        x_end = int(min(gray1.shape[1], x1 + 64))
        
        patch1 = gray1[y_start:y_end, x_start:x_end]
        
        # Corner coordinates relative to patch
        corners_coord = np.array([
            [x_start, y_start],
            [x_end, y_start],
            [x_start, y_end],
            [x_end, y_end]
        ], dtype=np.float32)
        
        # Extract corresponding patch from image2
        # For now, use center patch
        h2, w2 = gray2.shape
        patch2 = gray2[max(0, h2//2-64):min(h2, h2//2+64), max(0, w2//2-64):min(w2, w2//2+64)]
        
        best_patches.append((patch1, patch2, corners_coord))
    
    return best_patches[0]  # Return first patch for now


def warp_blend(img1, img2, H):
    """
    Warp image1 using homography H and blend with image2
    Similar to Phase1 implementation
    """
    # Get boundaries of warped image1
    h1, w1 = img1.shape[:2]
    corners1 = np.array([[0, 0], [w1, 0], [0, h1], [w1, h1]], dtype=np.float32).reshape(-1, 1, 2)
    corners1_warped = cv2.perspectiveTransform(corners1, H)
    
    # Get boundaries of image2
    h2, w2 = img2.shape[:2]
    corners2 = np.array([[0, 0], [w2, 0], [0, h2], [w2, h2]], dtype=np.float32).reshape(-1, 1, 2)
    
    # Combine boundaries
    all_corners = np.concatenate([corners1_warped, corners2], axis=0)
    
    # Compute output bounds
    x_min = int(np.floor(np.min(all_corners[:, 0, 0])))
    y_min = int(np.floor(np.min(all_corners[:, 0, 1])))
    x_max = int(np.ceil(np.max(all_corners[:, 0, 0])))
    y_max = int(np.ceil(np.max(all_corners[:, 0, 1])))
    
    # Translation to fit in positive quadrant
    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    
    # Warp image1
    output_size = (x_max - x_min, y_max - y_min)
    img1_warped = cv2.warpPerspective(img1, T @ H, output_size)
    
    # Composite image2
    img2_translated = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=img1.dtype)
    img2_translated[-y_min:-y_min+h2, -x_min:-x_min+w2] = img2
    
    # Blend: take image2 where it exists, else use warped image1
    mask = np.sum(img2_translated, axis=2) > 0
    result = img1_warped.copy()
    result[mask] = img2_translated[mask]
    
    return result


def stitch_panorama(images, model_path, model_type="Sup"):
    """
    Stitch a panorama from multiple images using deep learning homography
    """
    print(f"Loading model from {model_path}...")
    
    # Load model
    model = HomographyModel()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Model loaded successfully. Stitching {len(images)} images...")
    
    # Start with first image
    panorama = images[0]
    
    # Stitch remaining images
    for i in range(1, len(images)):
        print(f"Stitching image {i+1}/{len(images)}...")
        
        # Find best patches
        patch1, patch2, corners = find_best_patches(panorama, images[i])
        
        # Predict homography
        H4pt = predict_homography(model, patch1, patch2)
        
        # Convert to full homography matrix
        H = h4pt_to_homography(H4pt, corners)
        
        # Warp and blend
        panorama = warp_blend(panorama, images[i], H)
    
    return panorama


def main():
    # Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ImagePath', 
                       default='/Users/rohin/Documents/CV/AutoPano/Phase2/Data/Train/Set1', 
                       help='Path to image directory')
    Parser.add_argument('--ModelPath',
                       default='./checkpoints/0model.ckpt',
                       help='Path to trained model checkpoint')
    Parser.add_argument('--ModelType',
                       default='Sup',
                       choices=['Sup', 'Unsup'],
                       help='Model type: Sup (supervised) or Unsup (unsupervised)')
    Parser.add_argument('--OutputPath',
                       default='./mypano.png',
                       help='Path to save output panorama')
    
    Args = Parser.parse_args()
    
    """
    Read a set of images for Panorama stitching
    """
    images = []
    image_path = Args.ImagePath
    image_names = sorted([name for name in os.listdir(image_path) if name.endswith('.jpg')])
    
    print(f"Loading {len(image_names)} images from {image_path}...")
    for img_name in image_names:
        img_path = os.path.join(image_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            print(f"Loaded: {img_name} ({img.shape[1]}x{img.shape[0]})")
        else:
            print(f"Warning: Could not load {img_name}")
    
    if len(images) < 2:
        print("Error: Need at least 2 images for panorama stitching")
        return
    
    """
    Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
    """
    panorama = stitch_panorama(images, Args.ModelPath, Args.ModelType)
    
    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """
    if panorama is not None:
        # Crop black borders
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            panorama = panorama[y:y+h, x:x+w]
        
        print(f"Saving panorama to {Args.OutputPath}...")
        cv2.imwrite(Args.OutputPath, panorama)
        print(f"Panorama saved successfully as {Args.OutputPath}!")
    else:
        print("Error: Failed to create panorama")


if __name__ == "__main__":
    main()
