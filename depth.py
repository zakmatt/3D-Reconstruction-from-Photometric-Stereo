#!/usr/bin/env python3
from controllers import *
import numpy as np
from scipy.sparse import coo_matrix
import scipy
import scipy.misc, scipy.sparse
import scipy.sparse.linalg
from scipy import linalg

def get_depth(mask_image, normals_matrix):
    depth_matrix = -1.0 * np.ones(mask_image.shape)
    
    x_arr = []
    y_arr = []
    index = 0
    for (y, x), value in np.ndenumerate(mask_image):
        if value > 100.0:
            depth_matrix[y, x] = index
            y_arr.append(-y)
            x_arr.append(x)
            index += 1
            
    b, column, data, row = [], [], [], []
    idx = 0
    for (y, x), value in np.ndenumerate(depth_matrix):
        if value >= 0.0:
            normal = normals_matrix[y, x]/128.0 - 1.0
            normal /= np.linalg.norm(normal)
            
            if not np.isnan(np.sum(normal)):
                if np.abs(normal[2]) > 0.01:
                    # x values
                    index_matrix_val = depth_matrix[y, x + 1]
                    if index_matrix_val >= 0.0:
                        row.append(idx)
                        column.append(value)
                        data.append(-normal[2])
                        row.append(idx)
                        column.append(index_matrix_val)
                        data.append(normal[2])
                        b.append(-normal[0])
                        idx += 1
                        
                    # y values
                    index_matrix_val = depth_matrix[y - 1, x]
                    if index_matrix_val >= 0.0:
                        row.append(idx)
                        column.append(value)
                        data.append(-normal[2])
                        row.append(idx)
                        column.append(index_matrix_val)
                        data.append(normal[2])
                        b.append(-normal[1])
                        idx += 1
    
    matrix = coo_matrix((data, (row, column)), shape=(idx, index)).tocsc()
    b = np.array(b, dtype = np.float32)
    z = scipy.sparse.linalg.lsqr(matrix, b, iter_lim=1000)[0]
    
    index = 0
    for (y, x), value in np.ndenumerate(depth_matrix):
        if value >= 0.0:
            depth_matrix[y, x] = z[index]
            index += 1
            
    return depth_matrix

if __name__ == '__main__':
    normals_matrix = np.load('normals.npy')
    albedo_matrix = np.load('albedos.npy')
    images, mask_image = load_all_images('psmImages/buddha.txt')
    depth = get_depth(mask_image, normals_matrix)
    save_image(depth, 'depth.jpg')
    depth_2 = np.load('buddha.depths.dat')