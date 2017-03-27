#!/usr/bin/env python3
from controllers import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def load_all_images(file_name):
    with open(file_name, 'r') as file:
        number_of_files = int(file.readline().rstrip())
        images_paths = []
        for i in range(number_of_files):
            images_paths.append(file.readline().rstrip())
        images_mask = file.readline().rstrip()
    
    images = []
    for image_path in images_paths:
        images.append(open_image(image_path))

    mask_image = open_image(images_mask)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    
    return images, mask_image

def normals(images, mask_image, light):
    normal_map = np.zeros((mask_image.shape[0], mask_image.shape[1], 3))
    i_vec = np.zeros(len(images))
    
    for (y, x), value in np.ndenumerate(mask_image):
        if(value > 100.0):
            for pos, image in enumerate(images):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                i_vec[pos] = image[y, x]
                
            normal, _, _, _ = np.linalg.lstsq(light, i_vec)
            normal /= np.linalg.norm(normal)
            
            if not np.isnan(np.sum(normal)):
                normal_map[y, x] = normal
                          
    return normal_map

if __name__ == '__main__':
    light = np.loadtxt('light.txt', delimiter = ',')
    images, mask_image = load_all_images('psmImages/buddha.txt')
    normals_matrix = normals(images, mask_image, light)
    #normals_rescaled = rescale(normals_matrix)
    #normals_gray = cv2.cvtColor(normals_rescaled, cv2.COLOR_BGR2GRAY)
    plt.imshow(normals_matrix)
    plt.show()