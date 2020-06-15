import global_setting
import numpy as np
import cv2
import os
from colour_demosaicing import mosaicing_CFA_Bayer

def demosaic(mosaiced, pattern='RGGB', algo='Malvar2004'):
    """
    We will show the original image, then the filtered image, 
    then the results of three different demosaicing algorithms.
    """
    demosaic_func = global_setting.demosaic_func_dict[algo]    
    demosaiced = demosaic_func(mosaiced);
    return demosaiced

def mosaic(im, pattern='RGGB'):
    """
    im should have rgb as 012 channels. And have shapes as (height, width, 3)
    Returns an image of the bayer filter pattern.
    """    
    mosaiced = mosaicing_CFA_Bayer(im, pattern=pattern);
    return mosaiced

def rand_rgb_image(image_size, image_pattern):
    """
    Return image has RGB channels.
    """   
    if image_pattern == 'gaussian_rgb':
        red = np.random.normal(0.6, 0.3, (image_size, image_size, 1))
        blue = np.random.normal(0.5, 0.25, (image_size, image_size, 1))
        green = np.random.normal(0.9, 0.4, (image_size, image_size, 1))
        image = np.concatenate((red, green, blue), axis=2)
        image[image > 1.0] = 1.0
        image[image < 0.0] = 0.0
        image = (255 * image).astype('uint8')
    else:
        raise NotImplementedError()
    return image


