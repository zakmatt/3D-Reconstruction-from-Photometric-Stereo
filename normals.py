#!/usr/bin/env python3
from controllers import *
import cv2
import numpy as np
import matplotlib.pyplot as plt

def normals(images, mask_image, light):
    normals_matrix = np.zeros((mask_image.shape[0], mask_image.shape[1], 3))
    i_vec = np.zeros(len(images))
    
    for (y, x), value in np.ndenumerate(mask_image):
        if(value > 100.0):
            for pos, image in enumerate(images):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                i_vec[pos] = image[y, x]
                
            normal, _, _, _ = np.linalg.lstsq(light, i_vec)
            normal /= np.linalg.norm(normal)
            
            if not np.isnan(np.sum(normal)):
                normals_matrix[y, x] = normal
                          
    return normals_matrix

def albedo(images, mask_image, light, normals_matrix):
    albedo_matrix = np.zeros((mask_image.shape[0], mask_image.shape[1], 3), dtype = np.float32)
    i_vec = np.zeros((len(images), 3))
    
    for (y, x), value in np.ndenumerate(mask_image):
        if value > 100.0:
            for pos, image in enumerate(images):
                i_vec[pos] = image[y, x]
                
            I_trans = np.dot(light, normals_matrix[y, x])
            k = np.dot(np.transpose(i_vec), I_trans) / np.dot(I_trans, I_trans)
            
            if not np.isnan(np.sum(k)):
                albedo_matrix[y, x] = k
                             
    return albedo_matrix

if __name__ == '__main__':
    light = np.loadtxt('light.txt', delimiter = ',')
    images, mask_image = load_all_images('psmImages/buddha.txt')
    normals_matrix = normals(images, mask_image, light)
    albedo_matrix = albedo(images, mask_image, light, normals_matrix)
    save_image(albedo_matrix, 'albedo.jpg')
    save_image(normals_matrix, 'normals.jpg')
    np.save('normals', normals_matrix)
    np.save('albedos', albedo_matrix)