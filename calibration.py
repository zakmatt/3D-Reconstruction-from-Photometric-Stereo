#!/usr/bin/env python3
from controllers import open_image
import cv2
import numpy as np

# R = [0, 0, 1.0] based on the following page
# http://pages.cs.wisc.edu/~csverma/CS766_09/Stereo/stereo.html

with open('psmImages/chrome.txt', 'r') as file:
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

#calculate center of the ball
max_val = np.max(mask_image)
coords = np.argwhere(mask_image == max_val)
y_min = np.min(coords[:, 0])
y_max = np.max(coords[:, 0])
x_min = np.min(coords[:, 1])
x_max = np.max(coords[:, 1])

y_centre = (y_max + y_min)/2.0
x_centre = (x_max + x_min)/2.0
radius = (x_max - x_min)/2.0

R = [0, 0, 1.0]
L = []

for image in images:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    max_val = np.max(image)
    coords = np.argwhere(image == max_val)
    y_min = np.min(coords[:, 0])
    y_max = np.max(coords[:, 0])
    x_min = np.min(coords[:, 1])
    x_max = np.max(coords[:, 1])
    py = (y_max + y_min)/2.0
    px = (x_max + x_min)/2.0
    
    dx = px - x_centre
    dy = -(py - y_centre) # - to make it non-negative
    dz = np.sqrt(radius ** 2 - dx ** 2 - dy ** 2 )
    
    normal = np.array([dx, dy, dz])
    normal /= radius
    
    l = 2 * np.dot(normal, R) * normal - R
    l = np.round(l, decimals = 6)
    L.append(l)

L = np.array(L)
np.savetxt('light.txt', L, delimiter=',')