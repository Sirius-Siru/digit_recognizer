from skimage.feature import hog
from skimage import exposure

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