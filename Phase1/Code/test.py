import numpy as np
import cv2
import os
from skimage.feature import peak_local_max
from scipy.ndimage import maximum_filter

def detectCorners(imgs, choice):
    images = imgs.copy()
    print("detecting corners ...")
    detected_corners = []
    cmaps = []
    corner_images = []
    for i in images:
        image = i.copy()
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray_image = np.float32(gray_image)


        if(choice == 1):
            print("using Harris corner detection method.")
            corner_strength = cv2.cornerHarris(gray_image,2,3,0.001)
            corner_strength[corner_strength<0.01*corner_strength.max()] = 0
            detected_corner = np.where(corner_strength>0.0001*corner_strength.max())
            detected_corners.append(detected_corner)
            cmaps.append(corner_strength)
            image[corner_strength > 0.0001*corner_strength.max()]=[0,0,255]
            corner_images.append(image)
        else:
            print("using Shi-Tomashi corner detection method.")
            dst = cv2.goodFeaturesToTrack(gray_image, 1000 ,0.01, 10)
            dst = np.int0(dst)
            detected_corners.append(dst)
            for c in dst:
                x,y = c.ravel()
                cv2.circle(image,(x,y),3,(0, 0, 255),-1) 
                          
            corner_images.append(image)
            cmap = np.zeros(gray_image.shape) #not sure what to do
            cmaps.append(cmap)
    #filter detected corners
    #remove the corner one
    return detected_corners, cmaps, corner_images

def AdaptiveNonMaximalSuppression(images, C_maps, N_best):
    imgs = images.copy()
    anms_img = []
    anms_corners = []
    
    for i, img in enumerate(imgs):
        cmap = C_maps[i]
        
        # Find local maxima
        local_maximas = peak_local_max(cmap, min_distance=15)
        n_strong = local_maximas.shape[0]
        
        r = [np.Infinity for _ in range(n_strong)]
        x = np.zeros((n_strong, 1), dtype=int)
        y = np.zeros((n_strong, 1), dtype=int)
        eu_dist = 0

        # Compute suppression radius
        for i in range(n_strong):
            for j in range(n_strong):
                x_j = local_maximas[j][0]
                y_j = local_maximas[j][1]

                x_i = local_maximas[i][0]
                y_i = local_maximas[i][1]

                if cmap[x_j, y_j] > cmap[x_i, y_i]:
                    eu_dist = np.square(x_j - x_i) + np.square(y_j - y_i)
                if r[i] > eu_dist:
                    r[i] = eu_dist
                    x[i] = x_j
                    y[i] = y_j

        # Sort by suppression radius
        index = np.argsort(r)[::-1]  # Sort in descending order
        index = index[:N_best]
        
        # Initialize best coordinates
        N_best = min(N_best, x.shape[0])  # Handle cases where fewer maxima are found
        x_best = np.zeros((N_best, 1), dtype=int)
        y_best = np.zeros((N_best, 1), dtype=int)

        for i in range(N_best):
            x_best[i] = int(y[index[i]])  # Convert to Python int
            y_best[i] = int(x[index[i]])  # Convert to Python int
            cv2.circle(img, (int(x_best[i]), int(y_best[i])), 5, (0, 255, 0), -1)

        # Concatenate and append results
        anms_corner = np.hstack((x_best, y_best))
        anms_corners.append(anms_corner)
        anms_img.append(img)

    return anms_corners, anms_img


def main():
    img1 = cv2.imread("/Users/rohin/Documents/Computer Vision/AutoPano/Phase1/Data/Train/Set1/1.jpg")
    img2 = cv2.imread("/Users/rohin/Documents/Computer Vision/AutoPano/Phase1/Data/Train/Set1/2.jpg")
    img3 = cv2.imread("/Users/rohin/Documents/Computer Vision/AutoPano/Phase1/Data/Train/Set1/3.jpg")
    imgs = [img1, img2, img3]
    choice = 1
    detected_corners, cmaps, corner_images = detectCorners(imgs, choice)
    
    N_best = 500
    anms_corners, anms_img = AdaptiveNonMaximalSuppression(corner_images, cmaps, N_best)
    
    #Create a test output folder if it does not exist and store the corner images and the ANMS images
    if not os.path.exists("Test_Output"):
        os.makedirs("Test_Output")

    for i in range(len(corner_images)):
        cv2.imwrite("Test_Output/corners" + str(i+1) + ".png", corner_images[i])
        cv2.imwrite("Test_Output/anms" + str(i+1) + ".png", anms_img[i])

if __name__ == "__main__":
    main()