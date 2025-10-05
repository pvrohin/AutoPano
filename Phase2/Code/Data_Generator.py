#Load an image from Data folder
import cv2
import os
import random
import numpy as np

image_path = os.path.join('../Data/Train/1.jpg')
image = cv2.imread(image_path)
print(image.shape)

#Make the image of a standard size applicable to all images in the dataset
image = cv2.resize(image, (320, 240), interpolation = cv2.INTER_AREA)
print(image.shape)

#Display the image
# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

patch_size = 128  # MP × NP
pixel_shift_limit = 32  # ρ (maximum perturbation)
border_margin = 42  # Additional safety margin

h, w = image.shape[:2]  # M × N

# Calculate safe extraction bounds considering maximum perturbation
# The patch needs to be extracted from a region that, after maximum perturbation ρ,
# will still lie completely within the image boundaries
min_x = pixel_shift_limit  # Left boundary: at least ρ pixels from left edge
min_y = pixel_shift_limit  # Top boundary: at least ρ pixels from top edge
max_x = w - patch_size - pixel_shift_limit  # Right boundary: patch + ρ pixels from right edge
max_y = h - patch_size - pixel_shift_limit  # Bottom boundary: patch + ρ pixels from bottom edge

# Ensure we have valid bounds
if max_x <= min_x or max_y <= min_y:
    print("Error: Image too small for patch extraction with given perturbation")
    print(f"Required minimum size: {patch_size + 2 * pixel_shift_limit} × {patch_size + 2 * pixel_shift_limit}")
    print(f"Actual image size: {w} × {h}")
else:
    # Extract random patch from the safe region
    patch_x = random.randint(min_x, max_x)
    patch_y = random.randint(min_y, max_y)
    patch = image[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
    
    print(f"Patch extracted from: ({patch_x}, {patch_y}) to ({patch_x + patch_size}, {patch_y + patch_size})")
    print(f"Safe extraction region: x∈[{min_x}, {max_x}], y∈[{min_y}, {max_y}]")
    
    # Demonstrate why these bounds are necessary
    print(f"\nExplanation:")
    print(f"- Original patch corner: ({patch_x}, {patch_y})")
    print(f"- After maximum perturbation +ρ: ({patch_x + pixel_shift_limit}, {patch_y + pixel_shift_limit})")
    print(f"- After maximum perturbation -ρ: ({patch_x - pixel_shift_limit}, {patch_y - pixel_shift_limit})")
    print(f"- This ensures the warped patch stays within image bounds [0, {w}] × [0, {h}]")
    
    # Step 1: Define corner points of patch PA in image IA
    # Corner points: (top-left, top-right, bottom-left, bottom-right)
    pts_A = np.array([
        [patch_x, patch_y],                           # top-left
        [patch_x + patch_size, patch_y],              # top-right  
        [patch_x, patch_y + patch_size],              # bottom-left
        [patch_x + patch_size, patch_y + patch_size]  # bottom-right
    ], dtype=np.float32)
    
    print(f"\nOriginal corner points of PA:")
    print(f"Top-left: ({pts_A[0][0]}, {pts_A[0][1]})")
    print(f"Top-right: ({pts_A[1][0]}, {pts_A[1][1]})")
    print(f"Bottom-left: ({pts_A[2][0]}, {pts_A[2][1]})")
    print(f"Bottom-right: ({pts_A[3][0]}, {pts_A[3][1]})")
    
    # Step 2: Add random perturbation to corner points
    # Add both individual perturbations and a common translation
    common_translation_x = random.randint(-pixel_shift_limit, pixel_shift_limit)
    common_translation_y = random.randint(-pixel_shift_limit, pixel_shift_limit)
    
    pts_B = np.zeros_like(pts_A)
    
    for i in range(4):
        # Individual perturbation for each corner
        individual_perturbation_x = random.randint(-pixel_shift_limit, pixel_shift_limit)
        individual_perturbation_y = random.randint(-pixel_shift_limit, pixel_shift_limit)
        
        # Total perturbation = individual + common translation
        pts_B[i][0] = pts_A[i][0] + individual_perturbation_x + common_translation_x
        pts_B[i][1] = pts_A[i][1] + individual_perturbation_y + common_translation_y
    
    print(f"\nPerturbed corner points of PB:")
    print(f"Top-left: ({pts_B[0][0]}, {pts_B[0][1]})")
    print(f"Top-right: ({pts_B[1][0]}, {pts_B[1][1]})")
    print(f"Bottom-left: ({pts_B[2][0]}, {pts_B[2][1]})")
    print(f"Bottom-right: ({pts_B[3][0]}, {pts_B[3][1]})")
    print(f"Common translation: ({common_translation_x}, {common_translation_y})")
    
    # Step 3: Calculate homography H_AB from PA to PB
    H_AB = cv2.getPerspectiveTransform(pts_A, pts_B)
    print(f"\nHomography H_AB (PA -> PB):")
    print(H_AB)
    
    # Step 4: Calculate inverse homography H_BA = H_AB^(-1)
    H_BA = np.linalg.inv(H_AB)
    print(f"\nInverse homography H_BA (PB -> PA):")
    print(H_BA)
    
    # Step 5: Warp the entire image IA using H_AB to get image IB
    # H_AB transforms points from PA to PB, so it warps IA to IB
    image_B = cv2.warpPerspective(image, H_AB, (w, h))
    
    # Step 6: Extract patch PB from the same location in warped image IB
    # Since we used H_AB to warp the image, patch PA in IA becomes patch PB in IB
    patch_B = image_B[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
    
    print(f"\nExtracted patches:")
    print(f"Patch PA shape: {patch.shape}")
    print(f"Patch PB shape: {patch_B.shape}")
    
    # Verification: Check that the warped patch corners match our target
    print(f"\nVerification - Warped patch corners should match pts_B:")
    warped_corners = np.array([
        [patch_x, patch_y],
        [patch_x + patch_size, patch_y],
        [patch_x, patch_y + patch_size],
        [patch_x + patch_size, patch_y + patch_size]
    ], dtype=np.float32)
    
    # Transform these corners using H_AB
    warped_corners_homogeneous = np.hstack([warped_corners, np.ones((4, 1))])
    transformed_corners = (H_AB @ warped_corners_homogeneous.T).T
    transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:3]
    
    print(f"Original corners in IA: {warped_corners}")
    print(f"Transformed corners in IB: {transformed_corners}")
    print(f"Target corners (pts_B): {pts_B}")
    print(f"Match: {np.allclose(transformed_corners, pts_B, atol=1e-6)}")
    
    # Step 7: Calculate H4Pt (the 4-point homography representation)
    # H4Pt = CB - CA represents the displacement of the 4 corner points
    # This is the ground truth label for training the network
    H4Pt = (pts_B - pts_A).astype(np.float32)
    print(f"\nH4Pt (4-point homography ground truth):")
    print(f"Displacement vectors (CB - CA):")
    for i, (pt_a, pt_b) in enumerate(zip(pts_A, pts_B)):
        displacement = pt_b - pt_a
        print(f"Corner {i+1}: ({displacement[0]:.2f}, {displacement[1]:.2f})")
    
    # Step 8: Stack patches PA and PB depthwise to create input tensor
    # Input shape: MP × NP × 2K where K=3 for RGB images
    K = patch.shape[2] if len(patch.shape) == 3 else 1  # Number of channels
    
    if len(patch.shape) == 3:  # RGB image
        stacked_patches = np.dstack([patch, patch_B])  # Stack along depth axis
    else:  # Grayscale image
        stacked_patches = np.dstack([patch, patch_B])  # Stack along depth axis
    
    print(f"\nStacked patches shape: {stacked_patches.shape}")
    print(f"Expected shape: {patch_size} × {patch_size} × {2*K}")
    print(f"K (channels per patch): {K}")
    
    # Step 9: Flatten H4Pt to 1D array for training (8 values: 4 corners × 2 coordinates)
    H4Pt_flat = H4Pt.flatten()  # Shape: (8,)
    print(f"\nH4Pt flattened shape: {H4Pt_flat.shape}")
    print(f"H4Pt flattened values: {H4Pt_flat}")
    
    # Step 10: Prepare final training data
    print(f"\nFinal training data:")
    print(f"Input (stacked patches): {stacked_patches.shape}")
    print(f"Labels (H4Pt): {H4Pt_flat.shape}")
    print(f"Ground truth homography matrix H_AB shape: {H_AB.shape}")
    print(f"Ground truth homography matrix H_BA shape: {H_BA.shape}")
    
    # Step 11: Validation - Verify that H4Pt can reconstruct the homography
    # This is useful for understanding the relationship between 4-point and matrix representations
    print(f"\nValidation - Reconstructing homography from H4Pt:")
    print(f"H4Pt represents the displacement of 4 corner points")
    print(f"These 4 point correspondences can be used to compute the full homography matrix")
    
    # Optional: Show how to convert back from H4Pt to homography matrix
    reconstructed_H_AB = cv2.getPerspectiveTransform(pts_A, pts_A + H4Pt)
    print(f"Reconstructed H_AB from H4Pt matches original: {np.allclose(H_AB, reconstructed_H_AB, atol=1e-6)}")
    
    # Step 12: Training data summary
    print(f"\n" + "="*50)
    print(f"TRAINING DATA SUMMARY")
    print(f"="*50)
    print(f"Input tensor shape: {stacked_patches.shape}")
    print(f"  - Height: {stacked_patches.shape[0]} pixels")
    print(f"  - Width: {stacked_patches.shape[1]} pixels") 
    print(f"  - Channels: {stacked_patches.shape[2]} (2 patches × {K} channels each)")
    print(f"")
    print(f"Label tensor shape: {H4Pt_flat.shape}")
    print(f"  - 8 values representing (x,y) displacement for 4 corners")
    print(f"  - Order: [x1, y1, x2, y2, x3, y3, x4, y4]")
    print(f"  - Where (x1,y1) is top-left, (x2,y2) is top-right, etc.")
    print(f"")
    print(f"Network will learn: stacked_patches → H4Pt_flat")
    print(f"Then convert H4Pt_flat → full homography matrix for applications")
    print(f"="*50)

    # Step 13: Create visualization showing corner points
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Draw original patch corners in green
    for pt in pts_A:
        cv2.circle(vis_image, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    
    # Draw perturbed patch corners in red
    for pt in pts_B:
        cv2.circle(vis_image, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
    
    # Draw lines connecting corresponding corners
    for i in range(4):
        cv2.line(vis_image, 
                (int(pts_A[i][0]), int(pts_A[i][1])), 
                (int(pts_B[i][0]), int(pts_B[i][1])), 
                (255, 0, 0), 2)
    
    # Draw rectangle around original patch
    cv2.rectangle(vis_image, 
                 (int(pts_A[0][0]), int(pts_A[0][1])), 
                 (int(pts_A[3][0]), int(pts_A[3][1])), 
                 (0, 255, 0), 2)
    
    # Draw rectangle around perturbed patch
    cv2.rectangle(vis_image, 
                 (int(pts_B[0][0]), int(pts_B[0][1])), 
                 (int(pts_B[3][0]), int(pts_B[3][1])), 
                 (0, 0, 255), 2)

#Display the patches and visualization
    # Step 14: Demonstrate how this data would be used in training
    print(f"\n" + "="*50)
    print(f"TRAINING USAGE EXAMPLE")
    print(f"="*50)
    print(f"# In your training loop, you would use:")
    print(f"# X_train = stacked_patches  # Shape: (batch_size, 128, 128, 6)")
    print(f"# y_train = H4Pt_flat       # Shape: (batch_size, 8)")
    print(f"#")
    print(f"# model.fit(X_train, y_train)")
    print(f"#")
    print(f"# After training, to use the predicted H4Pt:")
    print(f"# predicted_H4Pt = model.predict(stacked_patches)")
    print(f"# predicted_H = cv2.getPerspectiveTransform(pts_A, pts_A + predicted_H4Pt.reshape(4,2))")
    print(f"="*50)

cv2.imshow('Patch PA (Original)', patch)
cv2.imshow('Patch PB (Warped)', patch_B)
cv2.imshow('Corner Point Visualization', vis_image)
cv2.waitKey(0)
cv2.destroyAllWindows()