# Train.py and Network_Supervised.py Connection

This document describes the changes made to connect `train.py` with `Network_Supervised.py` for homography estimation.

## Changes Made

### 1. Train.py
- **Import Fix**: Changed import from `Network.Network` to `Network.Network_Supervised`
- **Optimizer**: Added AdamW optimizer with learning rate 0.001 and weight decay 1e-4
- **Loss Function**: Imported and used `LossFn` from Network_Supervised.py
- **Data Preprocessing**: Updated `GenerateBatch` function to:
  - Convert images to grayscale
  - Resize to 128x128 pixels
  - Create 2-channel input (original + warped image)
  - Handle homography coordinates (8 values per sample)
- **Validation**: Fixed validation step call and simplified tensorboard logging

### 2. DataUtils.py
- **Label Reading**: Updated `ReadLabels` to reshape data into (num_samples, 8) for homography coordinates
- **Image Size**: Changed ImageSize to [128, 128, 2] for 2-channel homography input

### 3. Network_Supervised.py
- **Architecture**: The existing CNN architecture is suitable for 128x128 input
- **Output**: Network outputs 8 values for homography estimation
- **Loss Function**: Uses MSE loss for regression

## Usage

To run the training script:

```bash
cd Phase2/Code
python Train.py --BasePath /path/to/your/data --CheckPointPath ./checkpoints/ --ModelType Sup
```

## Data Format

The training script expects:
- Images in the specified BasePath directory
- Labels in `TxtFiles/LabelsTrain.txt` with 8 coordinates per line (flattened)
- Directory names in `TxtFiles/DirNamesTrain.txt`

## Model Architecture

The supervised model uses:
- 8 convolutional layers with BatchNorm and ReLU
- 3 max pooling layers
- 2 fully connected layers
- Dropout for regularization
- Output: 8 values for homography estimation

## Notes

- The current implementation duplicates the grayscale image as both channels for simplicity
- In a real implementation, you would have original and warped image pairs
- The model is designed for 128x128 input images with 2 channels
