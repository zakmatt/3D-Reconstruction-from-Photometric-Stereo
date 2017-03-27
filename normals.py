#!/usr/bin/env python3
from controllers import *
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    
def compute_albedo(light_matrix, mask_array, images_array, normal_map, threshold=100):
	shap = mask_array.shape
	shaper = (shap[0], shap[1], 3)

	albedo_map = np.zeros(shaper)
	ivec = np.zeros((len(images_array), 3))

	for (xT, value) in np.ndenumerate(mask_array):
		if(value > threshold):
			for (pos, image) in enumerate(images_array):
				ivec[pos] = image[xT[0], xT[1]]

			i_t = np.dot(light_matrix, normal_map[xT])

			k = np.dot(np.transpose(ivec), i_t)/(np.dot(i_t, i_t))

			if not np.isnan(np.sum(k)):
				albedo_map[xT] = k

	return albedo_map

if __name__ == '__main__':
    light = np.loadtxt('light.txt', delimiter = ',')
    images, mask_image = load_all_images('psmImages/buddha.txt')
    normals_matrix = normals(images, mask_image, light)
    albedo_matrix = albedo(images, mask_image, light, normals_matrix)
    save_image(albedo_matrix, 'albedo.jpg')
    save_image(normals_matrix, 'normals.jpg')
    np.save('normals', normals_matrix)
    np.save('albedos', albedo_matrix)