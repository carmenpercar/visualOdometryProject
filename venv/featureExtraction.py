import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def extract_features(image):
    option = 2

    if option == 1:
        # print("\nMethod: SIFT Scale Invariant Feature Transform\n")
        sift = cv.xfeatures2d.SIFT_create()
        kp = sift.detect(image, None)
        kp, des = sift.compute(image, kp)
    elif option == 2:
        # print("\nMethod: SURF Speeded-Up Robust Features\n")
        surf = cv.xfeatures2d.SURF_create()
        kp, des = surf.detectAndCompute(image, None)
    elif option == 3:
        print("\nMethod:ORB Gradient Location-Orientation Histogram\n")
        orb = cv.ORB_create()
        kp = orb.detect(image, None)
        des = orb.compute(image, kp)
    elif option == 4:
        print("\nMethod: FAST \n")
        fast = cv.FastFeatureDetector_create()
        kp = fast.detect(image, None)

    return kp, des


def visualize_features(image, kp):
    display = cv.drawKeypoints(image, kp, None)
    plt.imshow(display)
    plt.show()


def extract_features_dataset(images):
    kp_list = []
    des_list = []
    for image in images:
        kp, des = extract_features(image)
        kp_list.append(kp)
        des_list.append(des)

    return kp_list, des_list
