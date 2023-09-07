import numpy as np
import sys

from utils import *
from graphics import visualisation
from kabsch import kabsch_algorithm


def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    points = []
    for line in lines:
        x, y, z = map(float, line.strip().split())
        points.append([x, y, z])

    return np.array(points)


def test(all_points):

    P = all_points[0]
    Q = all_points[1]
    print("P:", *P, sep='\n')
    print("Q:", *Q, sep='\n')
    visualisation(P, Q)   # blue - P, red - Q

    P = center_points(P)
    Q = center_points(Q)

    optimal_rotation_matrix = kabsch_algorithm(P, Q)
    P_opt_rotated = (optimal_rotation_matrix @ P.T).T

    print("Opt. rotation matrix DET:", np.linalg.det(optimal_rotation_matrix))
    print("Optimal rotation matrix:", *optimal_rotation_matrix, sep='\n')
    print("P opt. rotated:", *P_opt_rotated, sep='\n')
    print("Q:", *Q, sep='\n')
    visualisation(P_opt_rotated, Q)  # blue - P opt. rotated, red - Q


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Filename input error.")
        sys.exit(1)

    all_points = []
    for filename in sys.argv[1:]:
        points = read_file(filename)
        all_points.append(points)

    test(all_points)
