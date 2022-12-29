import numpy as np
from skimage import io

def load_image(filename):
    img = io.imread(filename, as_gray=True) # use built-in grayscale loading
    crop = img[90:180, 54:124] # crop to face
    return crop
