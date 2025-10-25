#!/usr/bin/env python
"""
Fix all training setup issues
"""

import pandas as pd
import os
import numpy as np
import sys

def fix_training_setup():
    print("🔧 Fixing training setup for homography estimation...")
    
    # Paths
    synthetic_data_path = "/Users/rohin/Documents/CV/AutoPano/Phase2/Data/Train_synthetic"
    txt_files_path = "/Users/rohin/Documents/CV/AutoPano/Phase2/Code/TxtFiles"
    
    # Create txt_files directory if it doesn't exist
    os.makedirs(txt_files_path, exist_ok=True)
    
    print("1. Reading synthetic data...")
    try:
        labels_df = pd.read_csv(os.path.join(synthetic_data_path, "H4Pt_labels.csv"))
        image_names_df = pd.read_csv(os.path.join(synthetic_data_path, "image_names.csv"))
        print(f"   ✓ Loaded {len(labels_df)} samples")
    except Exception as e:
        print(f"   ❌ Failed to read synthetic data: {e}")
        return False
    
    print("2. Converting labels to homography format...")
    try:
        # Create directory names for the IA images (original images)
        dir_names = [f"Train_synthetic/IA/{name.replace('.jpg', '')}" for name in image_names_df['image_name']]
        
        # Create labels (flatten the 8 coordinates)
        labels = []
        for _, row in labels_df.iterrows():
            label_row = [row['corner_0_x'], row['corner_0_y'], 
                        row['corner_1_x'], row['corner_1_y'],
                        row['corner_2_x'], row['corner_2_y'],
                        row['corner_3_x'], row['corner_3_y']]
            labels.extend(label_row)
        
        print(f"   ✓ Converted {len(dir_names)} samples with 8 coordinates each")
    except Exception as e:
        print(f"   ❌ Failed to convert labels: {e}")
        return False
    
    print("3. Writing directory names...")
    try:
        with open(os.path.join(txt_files_path, "DirNamesTrain.txt"), "w") as f:
            for name in dir_names:
                f.write(name + "\n")
        print(f"   ✓ Wrote {len(dir_names)} directory names")
    except Exception as e:
        print(f"   ❌ Failed to write directory names: {e}")
        return False
    
    print("4. Writing labels...")
    try:
        with open(os.path.join(txt_files_path, "LabelsTrain.txt"), "w") as f:
            f.write(" ".join(map(str, labels)))
        print(f"   ✓ Wrote {len(labels)} coordinate values")
    except Exception as e:
        print(f"   ❌ Failed to write labels: {e}")
        return False
    
    print("5. Verifying data format...")
    try:
        # Test reading the data
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from Misc.DataUtils import ReadLabels, SetupDirNames
        
        train_labels = ReadLabels(os.path.join(txt_files_path, "LabelsTrain.txt"))
        train_dirs = SetupDirNames("/Users/rohin/Documents/CV/AutoPano/Phase2/Data")
        
        print(f"   ✓ Labels shape: {train_labels.shape}")
        print(f"   ✓ Directory count: {len(train_dirs)}")
        print(f"   ✓ First sample coordinates: {train_labels[0]}")
        
    except Exception as e:
        print(f"   ❌ Data verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 Training setup fixed successfully!")
    print("\nYou can now run training with:")
    print("python Train.py --BasePath /Users/rohin/Documents/CV/AutoPano/Phase2/Data --CheckPointPath ./checkpoints/ --ModelType Sup --NumEpochs 1 --MiniBatchSize 1")
    
    return True

if __name__ == "__main__":
    fix_training_setup()
