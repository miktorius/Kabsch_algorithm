import numpy as np


def calculate_deviation(points1, points2):
    s = 0
    for i in range(np.shape(points1)[0]):
        for j in range(3):
            s += (points1[i][j]-points2[i][j])**2
    return s


def center_points(points):
    points_array = np.array(points)
    centroid = np.mean(points_array, axis=0)
    centered_points = points_array - centroid

    return centered_points


def rotate_points(points: np.array, yaw, pitch, roll, apply_reflection):

    yaw_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                           [np.sin(yaw),  np.cos(yaw), 0],
                           [0,            0, 1]])

    pitch_matrix = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                             [0, 1,             0],
                             [-np.sin(pitch), 0, np.cos(pitch)]])

    roll_matrix = np.array([[1,            0,             0],
                            [0, np.cos(roll), -np.sin(roll)],
                            [0, np.sin(roll),  np.cos(roll)]])

    rotation_matrix = (np.dot(np.dot(roll_matrix, pitch_matrix), yaw_matrix))

    rotated_points = np.dot(rotation_matrix, points.T)

    # if (apply_reflection == True):
    #     reflection = np.array([[1, 0, 0],
    #                            [0, 1, 0],
    #                            [0, 0, -1]])
    #     rotated_points = reflection @ rotated_points

    return rotated_points.T, rotation_matrix


def add_noise(points: np.array, noise):

    mean = 0
    standard_deviation = noise

    noise = np.random.normal(mean, standard_deviation, size=points.shape)

    noisy_points = points + noise

    return noisy_points
