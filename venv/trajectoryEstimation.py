import numpy as np
import cv2 as cv


def estimate_motion(match, kp1, kp2, k):

    image1_points = []
    image2_points = []

    for m in match:
        point1 = kp1[m.queryIdx].pt
        point2 = kp2[m.trainIdx].pt
        image1_points.append(point1)
        image2_points.append(point2)

    M, _ = cv.findEssentialMat(np.array(image1_points), np.array(image2_points), k, method=cv.RANSAC)

    retval, R, t, _ = cv.recoverPose(M, np.array(image1_points), np.array(image2_points), k, 30)

    rmat = R

    tvec = t

    return rmat, tvec, image1_points, image2_points


def estimate_trajectory(matches, kp_list, k):

    trajectory = [np.array([0, 0, 0])]
    T = np.eye(4)

    for i, match in enumerate(matches):
        rmat, tvec, _, _ = estimate_motion(match, kp_list[i], kp_list[i + 1], k)
        Ti = np.eye(4)
        Ti[:3, :4] = np.c_[rmat.T, -rmat.T @ tvec]
        T = T @ Ti
        trajectory.append(T[:3, 3])
    trajectory = np.array(trajectory).T

    return trajectory
