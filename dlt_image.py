#!/usr/bin/python

import cv2
import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt


# DLT and homographies
# Write a program that will accept a left and right pair
# of stereo images each containing a checkerboard pattern.
# The program also accepts a point in the left image.
#
# The output will show a composite image with the left image
# on the left and a dot on the provided test point as well as the
# right image on the right with the corresponding
# point for the test point after a homography transformation.
#
# Arguments:
# left: Path to the left input image.
# right: Path to the right input image.
# cb_cols: The number of interior points per row in the
#          calibration pattern.
# cb_rows: The number of interior points per column in the
#          calibration pattern.
# point_x, point_y: Location of the test point in the left image.
# output: Path to the output image.


def main():
    if len(sys.argv) < 8:
        print
        'Usage: dlt_image.py left right cb_cols cb_rows point_x point_y output'
        sys.exit(1)

    left_filename = sys.argv[1]
    right_filename = sys.argv[2]
    cb_cols = int(sys.argv[3])
    cb_rows = int(sys.argv[4])
    point_x = float(sys.argv[5])
    point_y = float(sys.argv[6])
    output_filename = sys.argv[7]

    # Your code starts here
    #read images
    left_img = cv2.imread(left_filename)
    right_img = cv2.imread(right_filename)

    # define reference points, each row is x_left, y_left, x_right, y_right
    control_pts = np.array([[146, 63, 194, 54],
                            [183, 63, 223, 59],
                            [217, 65, 252, 65],
                            [251, 65, 282, 71],
                            [283, 66, 313, 77],
                            [316, 67, 346, 84],
                            [347, 69, 379, 91],
                            [377, 70, 414, 99],
                            [407, 71, 450, 106],
                            [435, 72, 488, 114],
                            [224, 313, 229, 290],
                            [258, 311, 256, 299],
                            [291, 308, 284, 309],
                            [322, 306, 314, 318],
                            [353, 303, 344, 328],
                            [384, 301, 375, 338],
                            [413, 298, 407, 349],
                            [441, 297, 440, 360],
                            [185, 171, 212, 158],
                            [220, 171, 240, 165],
                            [254, 170, 270, 173],
                            [286, 170, 301, 181],
                            [318, 169, 332, 189],
                            [350, 169, 363, 198],
                            [381, 168, 397, 207],
                            [409, 168, 432, 215]], dtype=np.float64)

    # normalize both left and right points
    mean = control_pts.mean(axis=0)
    ave_dis_left = ave_dis_right = 0
    for row in control_pts:
        ave_dis_left += math.sqrt((row[0] - mean[0]) ** 2 + (row[1] - mean[1]) ** 2)
        ave_dis_right += math.sqrt((row[2] - mean[2]) ** 2 + (row[3] - mean[3]) ** 2)
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
        row_a_1 = [0,0,0,-x_left,-y_left,-1,y_right*x_left,y_right*y_left,y_right]
        A.append(np.array(row_a_1).reshape((1, -1)))
        row_a_2 = [x_left,y_left,1,0,0,0,-x_right*x_left,-x_right*y_left,-x_right]
        A.append(np.array(row_a_2).reshape((1, -1)))
    A = np.vstack(A)

    #solve H
    u, s, vh = np.linalg.svd(A)
    H = vh[-1, :].reshape((3, -1))

    # Denormailize H
    H = np.dot(np.dot(np.linalg.inv(T_right), H), T_left)

    # compute corresponding point
    pt = np.array([point_x, point_y, 1]).reshape((3, 1))
    corres_pt = np.dot(H, pt)
    corres_pt[0] = corres_pt[0]/corres_pt[2]
    corres_pt[1] = corres_pt[1]/corres_pt[2]
    corres_pt[2] = 1

    # draw output image
    fig, axs = plt.subplots(1, 2)
    #draw left image
    cv2.circle(left_img, (int(point_x), int(point_y)), 3, (0, 0, 255), -1)
    axs[0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title('left image')
    #draw right image
    cv2.circle(right_img, (int(corres_pt[0]), int(corres_pt[1])), 3, (0, 0, 255), -1)
    axs[1].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title('right image')
    plt.savefig(output_filename)
    plt.show()


if __name__ == '__main__':
    main()
