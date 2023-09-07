import matplotlib.pyplot as plt
import numpy as np


def box_size(points):  # Scaling box for visualisation
    max_coord = -np.inf
    for each in points:
        for i in range(3):
            if abs(each[i]) > max_coord:
                max_coord = abs(each[i])
    return max_coord


def visualisation(points_set1, points_set2):

    max_coord = box_size(points_set2)
    left_lim, right_lim = -max_coord, max_coord

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(left_lim, right_lim)
    ax.set_ylim3d(left_lim, right_lim)
    ax.set_zlim3d(left_lim, right_lim)
    ax.set_box_aspect([1, 1, 1])

    # Represents the origin of coordinates
    ax.scatter(0, 0, 0, c='g', marker='^')
    for i in range(len(points_set1)):
        ax.scatter(points_set1[i][0], points_set1[i][1],
                   points_set1[i][2], c='b', marker='o')
        ax.scatter(points_set2[i][0], points_set2[i][1],
                   points_set2[i][2], c='r', marker='o')

    plt.show()
