#!/usr/bin/python

import cv2
import math
import numpy as np
import sys
import matplotlib.pyplot as plt

TEST_POINT = (150, 90)

REF_POINTS = [(130, 100),
              (140, 107),
              (150, 110),
              (160, 107),
              (170, 100)]


# DLT and homographies
# Write a program that will replicate the normalization DLT
# experiment shown in the slides.
#
# Calculate the homograph between two sets of points: the provided
# set of reference points and the reference points after corruption
# by Gaussian noise with mean 0 and standard deviation std.
# Calculate the homograhy n_interations times with different noise,
# either using or not using normalization. At each iteration,
# transform the provided test point and plot each outcome on the output
# image, along with the reference points.
#
# Arguments:
# std: The noise standard deviation.
# n_iterations: The number of iterations.
# output: Path to the output image.
# normalize: Either a 1 or a 0, to indicate whether normalization
#            should be used in the homography estimation.

def main():
    if len(sys.argv) < 5:
        print
        'Usage: dlt.py std n_iterations output normalize'
        sys.exit(1)

    std = float(sys.argv[1])
    n_iterations = int(sys.argv[2])
    output_filename = sys.argv[3]
    normalize = bool(int(sys.argv[4]))

    if normalize:
        print('With normalization')
    else:
        print('Without normalization')

    # Your code starts here
    reference_pts = np.array(REF_POINTS)
    # store projective points
    projective_pts = []
    for i in range(n_iterations):
        #original data
        reference_pts_1 = reference_pts.copy()
        #noisy data
        reference_pts_2 = np.random.normal(0, std, (reference_pts.shape[0], reference_pts.shape[1])) + reference_pts
        control_pts = np.hstack((reference_pts_1, reference_pts_2))
        #normalize
        if normalize:
            mean = control_pts.mean(axis=0)
            ave_dis_left = ave_dis_right = 0
            for pt in control_pts:
                ave_dis_left += math.sqrt((pt[0] - mean[0]) ** 2 + (pt[1] - mean[1]) ** 2)
                ave_dis_right += math.sqrt((pt[2] - mean[2]) ** 2 + (pt[3] - mean[3]) ** 2)
            ave_dis_left /= len(control_pts)
            ave_dis_right /= len(control_pts)
            T_left = np.array([[math.sqrt(2) / ave_dis_left, 0, -math.sqrt(2) * mean[0] / ave_dis_left],
                               [0, math.sqrt(2) / ave_dis_left, -math.sqrt(2) * mean[1] / ave_dis_left], [0, 0, 1]])
            T_right = np.array([[math.sqrt(2) / ave_dis_right, 0, -math.sqrt(2) * mean[2] / ave_dis_right],
                                [0, math.sqrt(2) / ave_dis_right, -math.sqrt(2) * mean[3] / ave_dis_right], [0, 0, 1]])
            for row in control_pts:
                tmp_left = np.dot(T_left, np.array([row[0], row[1], 1]).reshape((3, 1)))
                row[0] = tmp_left[0]
                row[1] = tmp_left[1]
                tmp_right = np.dot(T_right, np.array([row[2], row[3], 1]).reshape((3, 1)))
                row[2] = tmp_right[0]
                row[3] = tmp_right[1]

        # construct matrix A
        A = []
        for row in control_pts:
            x_left, y_left, x_right, y_right = row
            row_a_1 = [0, 0, 0, -x_left, -y_left, -1, y_right * x_left, y_right * y_left, y_right]
            A.append(np.array(row_a_1).reshape((1, -1)))
            row_a_2 = [x_left, y_left, 1, 0, 0, 0, -x_right * x_left, -x_right * y_left, -x_right]
            A.append(np.array(row_a_2).reshape((1, -1)))
        A = np.vstack(A)

        #solve H
        u, s, vh = np.linalg.svd(A)
        H = vh[-1, :].reshape((3, -1))
        if normalize:
            # Denormailize H
            H = np.dot(np.dot(np.linalg.inv(T_right), H), T_left)

        # compute corresponding point
        pt = np.array([TEST_POINT[0], TEST_POINT[1], 1]).reshape((3, 1))
        corres_pt = np.dot(H, pt)
        corres_pt[0] = corres_pt[0] / corres_pt[2]
        corres_pt[1] = corres_pt[1] / corres_pt[2]
        corres_pt[2] = 1
        #append the simulated projected point to the list
        projective_pts.append((corres_pt[0],corres_pt[1]))
    projective_pts = np.array(projective_pts)
    x = projective_pts[:,0]
    y = projective_pts[:,1]

    # draw projected points
    fig, ax = plt.subplots()
    plt.xlabel('X')
    plt.ylabel('Y')
    """set min and max value for axes"""
    ax.set_ylim([min(y)-10, max(y)+10])
    ax.set_xlim([min(x)-10, max(x)+10])
    #projected points
    plt.plot(x, y, 'o', color='b',label='Projected Points')
    #reference point
    plt.plot(TEST_POINT[0], TEST_POINT[1], 'o', color='r', label='Reference Point')
    plt.legend()
    if normalize:
        plt.title('Normalization')
    else:
        plt.title('No Normalization')
    plt.savefig(output_filename)
    plt.show()



if __name__ == '__main__':
    main()
