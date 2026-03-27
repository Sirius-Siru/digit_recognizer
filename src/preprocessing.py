from skimage.feature import hog
from skimage import exposure
from skimage.morphology import skeletonize
import cv2
import numpy as np
from joblib import Parallel, delayed

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
    angle = np.random.uniform(-5, 5)
    scale = np.random.uniform(0.9, 1.1)
    tx = np.random.uniform(-2, 2)  # shift ngang (pixel)
    ty = np.random.uniform(-2, 2)  # shift dọc

    M = cv2.getRotationMatrix2D((14, 14), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    return cv2.warpAffine(img, M, (28, 28), borderValue=0)

def skltn(img):
    img_bin = (img > 0.7).astype(np.uint8)
    skeleton = skeletonize(img_bin)
    return skeleton

def getDensityRatio(img){
    rows = np.any(img > 0, axis=1)
    cols = np.any(img > 0, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    cropped = img[ymin:ymax+1, xmin:xmax+1]

    height = cropped.shape[0]
    upper_half = cropped[:height//2, :]
    lower_half = cropped[height//2:, :]
    
    density_ratio = np.sum(upper_half) / (np.sum(lower_half) + 1e-6)
    return density_ratio
}

def aug_skltn_e_hog(img, is_train):
    # Augment
    aug = augment(img) if is_train else None 

    # Calculate half upper and half lower ratio
    origin_DRatio = getDensityRatio(img)
    aug_DRatio = getDensityRatio(aug) if is_train else None

    # Skeletonize
    sklt = skltn(img)
    aug_sklt = skltn(aug) if is_train else None

    # Hog
    hog_origin = extract_hog(img)
    hog_aug_origin = extract_hog(aug) if is_train else None

    # Combine
    orgin_combined = np.hstack((img.flatten(), hog_origin, sklt.flatten(), origin_DRatio))
    aug_combined = np.hstack((aug.flatten(), hog_aug_origin, aug_sklt.flatten(), aug_DRatio)) if is_train else None

    return np.vstack((orgin_combined, aug_combined)) if is_train else orgin_combined


def process_batch(batch_images, is_train):
    results = []
    for img in batch_images:
        results.append(aug_skltn_e_hog(img, is_train))
    return np.vstack(results)

def data_process(img, is_train):
    batch_size = 5000
    batches = [img[i:i + batch_size] for i in range(0, len(img), batch_size)]
    X_list = Parallel(n_jobs=-1)(delayed(process_batch)(b, is_train) for b in batches)

    return np.vstack(X_list)