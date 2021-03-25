import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import featureExtraction



def feature_matching(des1, des2):

    bf = cv.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    return matches


def filtering(matches, threshold):
    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append(m)
    return good


def filtered_feature_matches(des1, des2, threshold):

    matches = feature_matching(des1, des2)
    matches = filtering(matches, threshold)

    '''img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()'''
    return matches


def match_features_dataset(des_list, threshold):

    matches_dataset = []
    j = 0

    for j in range(len(des_list)-1):
        match = filtered_feature_matches(des_list[j], des_list[j + 1], threshold)
        matches_dataset.append(match)

    print(len(des_list))
    print(len(matches_dataset))
    return matches_dataset


def visualize_matches_dataset(images, kp_list, des_list):
    i = 0
    matches_dataset = match_features_dataset(des_list, 0.5)
    for i in range(len(matches_dataset)):
        img = cv.drawMatchesKnn(images[i], kp_list[i], images[i+1], kp_list[i+1], matches_dataset[i], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img)
        plt.show()
