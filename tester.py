import numpy as np
from utils import *
import math
from graphics import visualisation
from kabsch import kabsch_algorithm


def tester():

    points_quantity = np.random.randint(10, 20)
    points = np.random.randint(-25, 25, size=(points_quantity, 3))

    yaw = np.random.randint(0, 360)  # Z-axis
    pitch = np.random.randint(0, 360)  # Y-axis
    roll = np.random.randint(0, 360)  # X-axis

    apply_reflection = False  # np.random.randint(0, 2)
    noise = 0.1

    yaw = math.radians(yaw)
    pitch = math.radians(pitch)
    roll = math.radians(roll)

    rotated_points, rotation_matrix = rotate_points(
        points, yaw, pitch, roll, apply_reflection)
    noisy_points = add_noise(rotated_points, noise)
    points = center_points(points)
    noisy_points = center_points(noisy_points)

    print("Reflection applied:", True if apply_reflection else False)
    print("Points quantity:", points_quantity, '\n')
    print("Rotation matrix DET:", np.linalg.det(rotation_matrix))
    print("-------------Rotation matrix-------------", *rotation_matrix, sep='\n')
    print("-------------Original points-------------", *points, sep='\n')
    print("----------Rotated noisy points-----------",
          *noisy_points, '\n', sep='\n')
    visualisation(points, noisy_points)

    opt_rotation_matrix, is_reflection_applied, np_opt_rot_matrix = kabsch_algorithm(
        points, noisy_points)
    new_rotated_points = (opt_rotation_matrix @ points.T).T
    print("Kabsch rotation matrix DET:", np.linalg.det(opt_rotation_matrix))
    print("-----Kabsch optimal rotation matrix------",
          *opt_rotation_matrix, sep='\n')
    print("---------Original rotated points---------", *noisy_points, sep='\n')
    print("----------Kabsch rotated points----------",
          *new_rotated_points, sep='\n')
    print("RMSD:", calculate_deviation(
        noisy_points, new_rotated_points)/points_quantity)
    visualisation(noisy_points, new_rotated_points)


tester()
