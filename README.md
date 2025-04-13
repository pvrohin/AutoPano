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

## Pipeline Overview

1. **Corner Detection**
   - Detects corners in both images using either Harris or Shi-Tomasi corner detection
   - Output: Corner-marked images showing detected corners

2. **Adaptive Non-Maximal Suppression (ANMS)**
   - Selects the most distinctive corners while maintaining good spatial distribution
   - Output: Images with selected corners marked

3. **Feature Descriptor Extraction**
   - Extracts 8x8 feature descriptors around each corner
   - Applies Gaussian blur and normalization
   - Output: Feature vectors for each corner

4. **Feature Matching**
   - Matches features between images using ratio test
   - Output: Matched features visualization

5. **Homography Estimation**
   - Uses RANSAC to estimate the best homography matrix
   - Output: Best homography matrix and inlier matches

6. **Image Warping and Blending**
   - Warps and blends the images using the estimated homography
   - Output: Final stitched panorama

## Output Visualization

The program generates several output images showing the results of each step:

1. `corners_harris.jpg` or `corners_shi_tomasi.jpg`: Shows detected corners
2. `anms_corners.jpg`: Shows corners after ANMS
3. `feature_matches.jpg`: Shows matched features between images
4. `ransac_matches.jpg`: Shows inlier matches after RANSAC
5. `panorama.jpg`: Final stitched panorama

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