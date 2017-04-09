#!/usr/bin/env python3
from controllers import load_all_images, save_image
import numpy as np
from scipy.sparse import lil_matrix
import scipy
import scipy.misc, scipy.sparse
import scipy.sparse.linalg

def get_depth(mask_image, normals_matrix):
    rows, cols = np.where(mask_image>0.0)
    num_of_pixels = len(rows)
    full2obj = np.zeros(mask_image.shape, dtype = np.int)
    for pixel_pos in range(num_of_pixels):
        full2obj[rows[pixel_pos], cols[pixel_pos]] = pixel_pos
                
    M = lil_matrix((2 * num_of_pixels, num_of_pixels))
    u = np.empty((2 * num_of_pixels))
    
    for pixel_pos in range(num_of_pixels):
        h = rows[pixel_pos]
        w = cols[pixel_pos]
        
        nx = normals_matrix[h, w, 0]
        ny = normals_matrix[h, w, 1]
        nz = normals_matrix[h, w, 2]
        
        row_idx = (pixel_pos - 1) * 2 + 1
        if mask_image[h+1, w]:
            vertical_index = full2obj[h+1, w]
            u[row_idx] = ny
            M[row_idx, pixel_pos] = -nz
            M[row_idx, vertical_index] = nz
        elif mask_image[h-1, w]:
            vertical_index = full2obj[h-1, w]
            u[row_idx] = -ny
            M[row_idx, pixel_pos] = -nz
            M[row_idx, vertical_index] = nz
             
        # horizontal neighbors
        row_idx = (pixel_pos - 1) * 2 + 2
        if mask_image[h, w+1]:
            horizontal_index = full2obj[h, w+1]
            u[row_idx] = -nx
            M[row_idx, pixel_pos] = -nz
            M[row_idx, horizontal_index] = nz
        elif mask_image[h, w-1]:
            horizontal_index = full2obj[h, w-1]
            u[row_idx] = nx
            M[row_idx, pixel_pos] = -nz
            M[row_idx, horizontal_index] = nz
             
    u = M.transpose() * u
    M = M.transpose() * M
    z = scipy.sparse.linalg.lsqr(M, u)[0]
    
    Z = np.zeros(mask_image.shape, dtype = np.float32)
    for pixel_pos in range(num_of_pixels):
        h = rows[pixel_pos]
        w = cols[pixel_pos]
        Z[h, w] = z[pixel_pos]
        
    return Z

if __name__ == '__main__':
    normals_matrix = np.load('normals.npy')
    albedo_matrix = np.load('albedos.npy')
    images, mask_image = load_all_images('psmImages/horse.txt')
    depth = get_depth(mask_image, normals_matrix)
    save_image(depth, 'depth.jpg')
    np.save('depth', depth)