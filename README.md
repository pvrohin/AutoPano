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
python Wrapper.py --path to images
```

### Command Line Arguments

- `--path to images`: Path to the directory containing all the images  

## Pipeline Overview and Results

### 1. Corner Detection

Detects corners in both images using either Harris or Shi-Tomasi corner detection.

**Corner Detection Examples:**
- Image 1:  
  ![Harris Corners 1-2](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/corner_images_0.png?raw=true)
- Image 2:  
  ![Harris Corners 2-3](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/corner_images_1.png?raw=true)
- Image 3:  
  ![Harris Corners 1-3](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/corner_images_3.png?raw=true)

### 2. Adaptive Non-Maximal Suppression (ANMS)

Selects the most distinctive corners while maintaining good spatial distribution.

**ANMS Results:**
- Image 1:  
  ![ANMS Corners 1-2](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/anms_images_0.png?raw=true)
- Image 2:  
  ![ANMS Corners 2-3](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/anms_images_1.png?raw=true)
- Image 3:  
  ![ANMS Corners 1-3](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/anms_images_3.png?raw=true)

### 3. Feature Descriptor Extraction

Extracts 8x8 feature descriptors around each corner, applies Gaussian blur and normalization.

### 4. Feature Matching

Matches features between images using ratio test.

**Feature Matches:**
- Image 1 & 2:  
  ![Feature Matches 1-2](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/feature_matched_image_0.png?raw=true)
- Image 1 & 2 & 3:  
  ![Feature Matches 2-3](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/feature_matched_image_1.png?raw=true)

### 5. Homography Estimation

Uses RANSAC to estimate the best homography matrix.

**RANSAC Inlier Matches:**
- Image 1 & 2:  
  ![RANSAC Matches 1-2](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/ransac_image_0.png?raw=true)
- Image 1& 2 & 3:  
  ![RANSAC Matches 2-3](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/ransac_image_1.png?raw=true)

### 6. Image Warping and Blending

Warps and blends the images using the estimated homography.

**Final Panoramas:**
- Images 1 & 2:  
  ![Panorama 1-2](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/pano_image_0.png?raw=true)
- Complete Panorama (All three images):  
  ![Complete Panorama](https://github.com/pvrohin/AutoPano/blob/master/Phase1/Code/Results/Train/Set1/pano_image_1.png?raw=true)

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
