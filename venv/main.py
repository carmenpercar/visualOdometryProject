import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from DataHandler import *
import featureExtraction
import featureMatching
import trajectoryEstimation


dataset_handler = DataHandler()
images = dataset_handler.images


i = 0

kp_list, des_list = featureExtraction.extract_features_dataset(images)

matches = featureMatching.match_features_dataset(des_list, threshold=0.5)

match = matches[i]
kp1 = kp_list[i]
kp2 = kp_list[i+1]
k = dataset_handler.k
depth = dataset_handler.depth_map[i]

rmat, tvec, image1_points, image2_points = trajectoryEstimation.estimate_motion(match, kp1, kp2, k)

print("Estimated rotation:\n {0}".format(rmat))
print("Estimated translation:\n {0}".format(tvec))

image1 = dataset_handler.images[i]
image2 = dataset_handler.images[i + 1]
# image1 = dataset_handler.images_rgb[i]
# image2 = dataset_handler.images_rgb[i + 1]

'''image_move = visualize_camera_movement(image1, image1_points, image2, image2_points)
plt.figure(figsize=(16, 12), dpi=100)
plt.imshow(image_move)
'''
trajectory = trajectoryEstimation.estimate_trajectory(matches, kp_list, k)

visualize_trajectory(trajectory)
