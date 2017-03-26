#!/usr/bin/env python3
from controllers import open_image
import cv2

with open('psmImages/buddha.txt', 'r') as file:
    number_of_files = int(file.readline().rstrip())
    images_paths = []
    for i in range(number_of_files):
        images_paths.append(file.readline().rstrip())
    images_mask = file.readline().rstrip()
    
images = []
for image_path in images_paths:
    images.append(open_image(image_path))