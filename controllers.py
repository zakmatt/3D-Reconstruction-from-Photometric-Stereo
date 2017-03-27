#!/usr/bin/env python3
import cv2
import numpy as np
from PIL import Image

def rescale(image):
    image = image.astype('float32')
    current_min = np.min(image)
    current_max = np.max(image)
    image = (image - current_min)/(current_max - current_min) * 255
    return image

def open_image(image_path):
    image = Image.open(image_path)
    image = np.array(image, dtype = np.float32)
    #image = np.swapaxes(image, 0, 2)
    #image = np.swapaxes(image, 1, 2)
    return image

def save_image(image, path):
    image = rescale(image)
    image = np.array(image, dtype = np.uint8)
    cv2.imwrite(path, image)
    
if __name__=='__main__':
    pass