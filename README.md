# MyAutoPano: Image Stitching Project

This project implements an automatic image stitching pipeline that combines multiple images into a panoramic view. The implementation uses classical computer vision techniques including corner detection, feature matching, and homography estimation.

## Features

- Harris and Shi-Tomasi corner detection
- Adaptive Non-Maximal Suppression (ANMS) for corner selection
- Feature descriptor extraction using 8x8 patches
- Feature matching with ratio test
- RANSAC-based homography estimation
- Image warping and blending

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- scikit-image
- SciPy

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd AutoPano/Phase1/Code
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python Wrapper.py --image1 path/to/image1.jpg --image2 path/to/image2.jpg
```

### Command Line Arguments

- `--image1`: Path to the first input image
- `--image2`: Path to the second input image
- `--use_harris`: Use Harris corner detection (default: True)
- `--block_size`: Block size for corner detection (default: 2)
- `--ksize`: Aperture parameter for corner detection (default: 3)
- `--k`: Harris detector free parameter (default: 0.04)
- `--num_corners`: Number of corners to detect (default: 1000)
- `--N_best`: Number of best corners to keep after ANMS (default: 400)

## Pipeline Overview and Results

### 1. Corner Detection
Detects corners in both images using either Harris or Shi-Tomasi corner detection.

**Harris Corner Detection Examples:**
- Image 1 & 2:
![Harris Corners 1-2]([results/corners_harris_1_2.jpg](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/corner_images_0.png))
- Image 2 & 3:
![Harris Corners 2-3]([results/corners_harris_2_3.jpg](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/corner_images_1.png))
- Image 1 & 3:
![Harris Corners 1-3]([results/corners_harris_1_3.jpg](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/corner_images_2.png))

**Shi-Tomasi Corner Detection Examples:**
- Image 1 & 2:
![Shi-Tomasi Corners 1-2](results/corners_shi_tomasi_1_2.jpg)
- Image 2 & 3:
![Shi-Tomasi Corners 2-3](results/corners_shi_tomasi_2_3.jpg)
- Image 1 & 3:
![Shi-Tomasi Corners 1-3](results/corners_shi_tomasi_1_3.jpg)

### 2. Adaptive Non-Maximal Suppression (ANMS)
Selects the most distinctive corners while maintaining good spatial distribution.

**ANMS Results:**
- Image 1 & 2:
![ANMS Corners 1-2]([results/anms_corners_1_2.jpg](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/anms_images_0.png))
- Image 2 & 3:
![ANMS Corners 2-3]([results/anms_corners_2_3.jp](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/anms_images_1.png)g)
- Image 1 & 3:
![ANMS Corners 1-3]([results/anms_corners_1_3.jpg](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/anms_images_2.png))

### 3. Feature Descriptor Extraction
Extracts 8x8 feature descriptors around each corner, applies Gaussian blur and normalization.

### 4. Feature Matching
Matches features between images using ratio test.

**Feature Matches:**
- Image 1 & 2:
![Feature Matches 1-2]([results/feature_matches_1_2.jpg](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/feature_matched_image_0.png))
- Image 2 & 3:
![Feature Matches 2-3]([results/feature_matches_2_3.jpg](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/feature_matched_image_1.png))

### 5. Homography Estimation
Uses RANSAC to estimate the best homography matrix.

**RANSAC Inlier Matches:**
- Image 1 & 2:
![RANSAC Matches 1-2]([results/ransac_matches_1_2.jpg](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/ransac_image_0.png))
- Image 2 & 3:
![RANSAC Matches 2-3]([results/ransac_matches_2_3.jpg](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/ransac_image_1.png))

### 6. Image Warping and Blending
Warps and blends the images using the estimated homography.

**Final Panoramas:**
- Images 1 & 2:
![Panorama 1-2]([results/panorama_1_2.jpg](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/pano_image_0.png))
- Images 2 & 3:
![Panorama 2-3]([results/panorama_2_3.jpg](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/pano_image_1.png))
- Complete Panorama (All three images):
![Complete Panorama]([results/complete_panorama.jpg](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/corner_images_1.png))

## Output Files

The program generates the following output files in the `results` directory for each image pair:

1. `corners_harris_X_Y.jpg` or `corners_shi_tomasi_X_Y.jpg`: Shows detected corners for images X and Y
2. `anms_corners_X_Y.jpg`: Shows corners after ANMS for images X and Y
3. `feature_matches_X_Y.jpg`: Shows matched features between images X and Y
4. `ransac_matches_X_Y.jpg`: Shows inlier matches after RANSAC for images X and Y
5. `panorama_X_Y.jpg`: Stitched panorama for images X and Y
6. `complete_panorama.jpg`: Final panorama combining all three images

## Example

```bash
python Wrapper.py --image1 images/left.jpg --image2 images/right.jpg --use_harris True
```

This will process the two images and generate the output files showing each step of the pipeline.

## Notes

- The images should have sufficient overlap for successful stitching
- For best results, use images taken from the same viewpoint with minimal perspective changes
- The default parameters work well for most cases, but you may need to adjust them for specific scenarios

## Troubleshooting

1. **Not enough matches found**
   - Try adjusting the corner detection parameters
   - Ensure images have sufficient overlap
   - Try using different corner detection methods

2. **Poor stitching results**
   - Check if the homography estimation is accurate
   - Verify that the feature matching is working correctly
   - Consider adjusting the RANSAC parameters

## License

[Your License Here]

## Acknowledgments

This project was developed as part of the RBE/CS Fall 2022 course on Classical and Deep Learning Approaches for Geometric Computer Vision at Worcester Polytechnic Institute.
