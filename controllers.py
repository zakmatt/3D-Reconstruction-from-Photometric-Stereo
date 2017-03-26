#!/usr/bin/env python3
import numpy as np
from PIL import Image

def open_image(image_path):
    image = Image.open(image_path)
    image = np.array(image, dtype = np.float32)
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)
    return image