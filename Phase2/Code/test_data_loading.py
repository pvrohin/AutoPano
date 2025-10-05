import numpy as np
import pandas as pd
import cv2
import os

def test_data_loading():
    """Test script to verify the generated dataset can be loaded correctly."""
    
    # Test loading training data
    train_path = '../Data/Train_synthetic/'
    
    if os.path.exists(os.path.join(train_path, 'stacked_patches.npy')):
        print("Loading training data...")
        
        # Load stacked patches
        stacked_patches = np.load(os.path.join(train_path, 'stacked_patches.npy'))
        print(f"Stacked patches shape: {stacked_patches.shape}")
        
        # Load H4Pt labels
        H4Pt_labels = np.load(os.path.join(train_path, 'H4Pt_labels.npy'))
        print(f"H4Pt labels shape: {H4Pt_labels.shape}")
        
        # Load image names
        image_names = pd.read_csv(os.path.join(train_path, 'image_names.csv'))
        print(f"Number of samples: {len(image_names)}")
        
        # Display first sample
        print(f"\nFirst sample:")
        print(f"Image name: {image_names.iloc[0]['image_name']}")
        print(f"Stacked patch shape: {stacked_patches[0].shape}")
        print(f"H4Pt values: {H4Pt_labels[0]}")
        
        # Show the patches
        first_patch = stacked_patches[0]
        patch_A = first_patch[:, :, :3]  # First 3 channels
        patch_B = first_patch[:, :, 3:]  # Last 3 channels
        
        print(f"Patch A shape: {patch_A.shape}")
        print(f"Patch B shape: {patch_B.shape}")
        
        # Display patches
        cv2.imshow('Patch A (Original)', patch_A)
        cv2.imshow('Patch B (Warped)', patch_B)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("Data loading test completed successfully!")
        
    else:
        print("No training data found. Please run Data_Generator.py first.")

if __name__ == '__main__':
    test_data_loading()
