import numpy as np
import pandas as pd
import skimage
from skimage.transform import resize
import tensorflow as tf
import keras

def _norm(_image):
    """a function to standardize the pixel intensity ranges 
    """
    _image *= 1.0/_image.max() 
    return _image


def process_X(img_list):
    """A function to prepare images for classification 
    
    NOTE: this is model dependent
    """
    # stub a color channel
    # and standardize the size (will not preseve the aspect ratio)
    # prolly a bad idea
    
    # authors use 320x320
    imgs = [skimage.color.gray2rgb(resize(_norm(i), (320, 320))) for i in img_list]
#     process = lambda x : tf.keras.applications.resnet_v2.preprocess_input(x)
#     imgs = list(map(process, imgs))
    imgs = np.asarray(imgs)
    return imgs



def process_Y(data, target_var):
    """A function to process the dependenr variable according to the U-Zeroes
    framework:

        'U-Zeroes: We map all instances of the uncertain label to 0.'
    """
    target = data[target_var]
    target = np.where(target < 0, 0, target)
    target = keras.utils.to_categorical(target, 2)
    
    return target