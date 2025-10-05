import cv2
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_patch_pair(image, patch_size=128, pixel_shift_limit=32, border_margin=42):
    """
    Generate a pair of patches (PA, PB) and their homography from a single image.
    
    Args:
        image: Input image
        patch_size: Size of patches to extract
        pixel_shift_limit: Maximum perturbation radius
        border_margin: Safety margin from image boundaries
    
    Returns:
        stacked_patches: Stacked patches PA and PB (MP × NP × 2K)
        H4Pt_flat: Flattened 4-point homography (8 values)
        success: Boolean indicating if generation was successful
    """
    h, w = image.shape[:2]  # M × N
    
    # Calculate safe extraction bounds considering maximum perturbation
    min_x = pixel_shift_limit
    min_y = pixel_shift_limit
    max_x = w - patch_size - pixel_shift_limit
    max_y = h - patch_size - pixel_shift_limit
    
    # Check if image is large enough
    if max_x <= min_x or max_y <= min_y:
        return None, None, False
    
    # Extract random patch from the safe region
    patch_x = random.randint(min_x, max_x)
    patch_y = random.randint(min_y, max_y)
    patch = image[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
    
    # Define corner points of patch PA
    pts_A = np.array([
        [patch_x, patch_y],                           # top-left
        [patch_x + patch_size, patch_y],              # top-right  
        [patch_x, patch_y + patch_size],              # bottom-left
        [patch_x + patch_size, patch_y + patch_size]  # bottom-right
    ], dtype=np.float32)
    
    # Add random perturbation to corner points
    common_translation_x = random.randint(-pixel_shift_limit, pixel_shift_limit)
    common_translation_y = random.randint(-pixel_shift_limit, pixel_shift_limit)
    
    pts_B = np.zeros_like(pts_A)
    for i in range(4):
        individual_perturbation_x = random.randint(-pixel_shift_limit, pixel_shift_limit)
        individual_perturbation_y = random.randint(-pixel_shift_limit, pixel_shift_limit)
        
        pts_B[i][0] = pts_A[i][0] + individual_perturbation_x + common_translation_x
        pts_B[i][1] = pts_A[i][1] + individual_perturbation_y + common_translation_y
    
    # Calculate homography and warp image
    H_AB = cv2.getPerspectiveTransform(pts_A, pts_B)
    image_B = cv2.warpPerspective(image, H_AB, (w, h))
    
    # Extract patch PB from warped image
    patch_B = image_B[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
    
    # Calculate H4Pt (4-point homography)
    H4Pt = (pts_B - pts_A).astype(np.float32)
    H4Pt_flat = H4Pt.flatten()
    
    # Stack patches depthwise
    K = patch.shape[2] if len(patch.shape) == 3 else 1
    stacked_patches = np.dstack([patch, patch_B])
    
    return stacked_patches, H4Pt_flat, True

def generate_dataset(data_type='Train', num_patches_per_image=5):
    """
    Generate dataset for a specific data type (Train/Val).
    
    Args:
        data_type: 'Train' or 'Val'
        num_patches_per_image: Number of patches to generate per image
    
    Returns:
        all_stacked_patches: List of stacked patches
        all_H4Pt: List of H4Pt labels
        all_image_names: List of corresponding image names
    """
    print(f"Generating {data_type} dataset...")
    
    # Set up paths
    if data_type == 'Train':
        image_path = '../Data/Train/'
        save_path = '../Data/Train_synthetic/'
        num_images = 5000  # Adjust based on your dataset
    else:  # Val
        image_path = '../Data/Val/'
        save_path = '../Data/Val_synthetic/'
        num_images = 1000  # Adjust based on your dataset
    
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + 'PA/', exist_ok=True)
    os.makedirs(save_path + 'PB/', exist_ok=True)
    os.makedirs(save_path + 'IA/', exist_ok=True)
    
    all_stacked_patches = []
    all_H4Pt = []
    all_image_names = []
    
    successful_patches = 0
    failed_patches = 0
    
    # Process each image
    for img_idx in tqdm(range(1, num_images + 1), desc=f"Processing {data_type} images"):
        image_file = f'{img_idx}.jpg'
        image_path_full = os.path.join(image_path, image_file)
        
        if not os.path.exists(image_path_full):
            continue
            
        # Load and resize image
        image = cv2.imread(image_path_full)
        if image is None:
            continue
            
        image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
        
        # Generate multiple patches per image
        for patch_idx in range(num_patches_per_image):
            stacked_patches, H4Pt_flat, success = generate_patch_pair(image)
            
            if success:
                # Save individual patches
                patch_name = f'{img_idx}_{patch_idx}.jpg'
                cv2.imwrite(os.path.join(save_path, 'PA', patch_name), stacked_patches[:, :, :3])
                cv2.imwrite(os.path.join(save_path, 'PB', patch_name), stacked_patches[:, :, 3:])
                cv2.imwrite(os.path.join(save_path, 'IA', patch_name), image)
                
                # Store data
                all_stacked_patches.append(stacked_patches)
                all_H4Pt.append(H4Pt_flat)
                all_image_names.append(patch_name)
                
                successful_patches += 1
            else:
                failed_patches += 1
    
    print(f"Generated {successful_patches} successful patches, {failed_patches} failed patches")
    
    # Save data
    if all_stacked_patches:
        # Save stacked patches as numpy array
        stacked_patches_array = np.array(all_stacked_patches)
        np.save(os.path.join(save_path, 'stacked_patches.npy'), stacked_patches_array)
        
        # Save H4Pt labels
        H4Pt_array = np.array(all_H4Pt)
        np.save(os.path.join(save_path, 'H4Pt_labels.npy'), H4Pt_array)
        
        # Save as CSV for easy loading
        df_H4Pt = pd.DataFrame(H4Pt_array, columns=[f'corner_{i//2}_{"x" if i%2==0 else "y"}' for i in range(8)])
        df_H4Pt.to_csv(os.path.join(save_path, 'H4Pt_labels.csv'), index=False)
        
        # Save image names
        df_names = pd.DataFrame(all_image_names, columns=['image_name'])
        df_names.to_csv(os.path.join(save_path, 'image_names.csv'), index=False)
        
        print(f"Saved {len(all_stacked_patches)} patch pairs to {save_path}")
        print(f"Stacked patches shape: {stacked_patches_array.shape}")
        print(f"H4Pt labels shape: {H4Pt_array.shape}")
    
    return all_stacked_patches, all_H4Pt, all_image_names

def main():
    """Main function to generate complete dataset."""
    print("Starting dataset generation...")
    
    # Generate training data
    train_patches, train_labels, train_names = generate_dataset('Train', num_patches_per_image=5)
    
    # Generate validation data
    val_patches, val_labels, val_names = generate_dataset('Val', num_patches_per_image=5)
    
    print("\nDataset generation complete!")
    print(f"Training data: {len(train_patches)} patch pairs")
    print(f"Validation data: {len(val_patches)} patch pairs")

if __name__ == '__main__':
    main()