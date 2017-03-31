#!/usr/bin/env python3
from controllers import *
import cv2
import numpy as np
    

def normals(images, mask_image, light):
    normals_matrix = np.zeros((mask_image.shape[0], mask_image.shape[1], 3))
    normals_matrix[:, :, 2] = 1
    I = np.zeros(len(images))
    
    for (y, x), value in np.ndenumerate(mask_image):
        if value > 100.0:
            for pos, image in enumerate(images):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                I[pos] = image[y, x]
                
            #[dx, dy, dz]    
            normal, _, _, _ = np.linalg.lstsq(light, I)
            normal /= np.linalg.norm(normal)
            
            if not np.isnan(np.sum(normal)):
                normals_matrix[y, x] = normal
                          
    return normals_matrix

def albedo(images, mask_image, light, normals_matrix):
    albedo_matrix = np.zeros((mask_image.shape[0], mask_image.shape[1], 3), dtype = np.float32)
    I = np.zeros(len(images))
    number_of_channels = 3
    
    for channel in range(number_of_channels):
        for (y, x), value in np.ndenumerate(mask_image):
            if value > 100.0:
                for pos, image in enumerate(images):
                    I[pos] = image[y, x, channel]
                    
                J = np.dot(light, normals_matrix[y, x])
                k = np.dot(np.transpose(I), J) / np.dot(J, J)
                
                if not np.isnan(np.sum(k)):
                    # 2 - channel switches from GBR to RGB
                    albedo_matrix[y, x, 2 - channel] = k
                             
    return albedo_matrix

if __name__ == '__main__':
    light = np.loadtxt('light.txt', delimiter = ',')
    images, mask_image = load_all_images('psmImages/cat.txt')
    normals_matrix = normals(images, mask_image, light)
    albedo_matrix = albedo(images, mask_image, light, normals_matrix)
    save_image(albedo_matrix, 'albedo.jpg')
    save_image(normals_matrix, 'normals.jpg')
    np.save('normals', normals_matrix)
    np.save('albedos', albedo_matrix)
