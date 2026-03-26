from skimage.feature import hog
from skimage import exposure
from skimage.morphology import skeletonize
import cv2
import numpy as np

def extract_hog(img_28x28):
    features, hog_image = hog(
        img_28x28,
        orientations=9,      
        pixels_per_cell=(4, 4), 
        cells_per_block=(2, 2), 
        visualize=True,      
        channel_axis=None
    )
    return features

def augment(img):
    angle = np.random.uniform(-10, 10)
    scale = np.random.uniform(0.9, 1.1)
    tx = np.random.uniform(-2, 2)  # shift ngang (pixel)
    ty = np.random.uniform(-2, 2)  # shift dọc

    M = cv2.getRotationMatrix2D((14, 14), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    return cv2.warpAffine(img, M, (28, 28), borderValue=0)

def skltn(img):
    img_bin = (img > 0.5).astype(np.uint8)
    skeleton = skeletonize(img_bin)
    return skeleton