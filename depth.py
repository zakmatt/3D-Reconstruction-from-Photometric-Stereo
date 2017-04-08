#!/usr/bin/env python3
from controllers import *
import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
import scipy
import scipy.misc, scipy.sparse
from scipy.sparse.linalg import spsolve
import scipy.sparse.linalg
import scipy.sparse as sp
from matplotlib import pyplot as plt

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
            #normal = normals_matrix[y, x]/128.0 - 1.0
            #normal /= np.linalg.norm(normal)
            
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
    #b = np.transpose(matrix) * b
    #matrix = np.transpose(matrix) * matrix
    z = scipy.sparse.linalg.lsqr(matrix, b, iter_lim=1000)[0]
    
    index = 0
    for (y, x), value in np.ndenumerate(depth_matrix):
        if value >= 0.0:
            depth_matrix[y, x] = z[index]
            index += 1
            
    return depth_matrix

def get_depth_3(mask_image, normals_matrix):
    rows, cols = np.where(mask_image>0.0)
    num_of_pixels = len(rows)
    full2obj = np.zeros(mask_image.shape, dtype = np.int)
    for pixel_pos in range(num_of_pixels):
        full2obj[rows[pixel_pos], cols[pixel_pos]] = pixel_pos
                
    M = lil_matrix((2 * num_of_pixels, num_of_pixels))
    #u = lil_matrix((2 * num_of_pixels, 1))
    #M = np.empty((2 * num_of_pixels, num_of_pixels))
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

def display_depth_matplotlib(z):
    """
    Same as above but using matplotlib instead.
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import LightSource
    
    m, n = z.shape
    x, y = np.mgrid[0:m, 0:n]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ls = LightSource(azdeg=0, altdeg=65)
    greyvals = ls.shade(z, plt.cm.Greys)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=False, facecolors=greyvals)
    plt.axis('off')
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    normals_matrix = np.load('normals.npy')
    albedo_matrix = np.load('albedos.npy')
    images, mask_image = load_all_images('psmImages/cat.txt')
    #depth = unbiased_integrate(normals_matrix[:,:,0], normals_matrix[:,:,1], normals_matrix[:,:,2], mask_image)
    depth_2 = get_depth(mask_image, normals_matrix)
    #save_image(depth, 'depth.jpg')
    save_image(Z, 'depth_2.jpg')