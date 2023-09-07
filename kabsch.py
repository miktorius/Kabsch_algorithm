import numpy as np

from utils import *
from svd import golub_reinsch_svd


def kabsch_algorithm(P, Q):

    covariance_matrix = P.T @ Q
    U, _, V = golub_reinsch_svd(covariance_matrix)

    # print("S:", *S, sep='\n')
    # print("U:", *U, sep='\n')
    # print("V:", *V, sep='\n')

    # Ub, _, VTb = np.linalg.svd(covariance_matrix)

    # print("Sb:", *Sb)
    # print("Ub:", *Ub, sep='\n')
    # print("VTb.T:", *VTb.T, sep='\n')W

    # Reflection detector
    d = np.sign(np.linalg.det(V @ U.T))
    coord_system_corrector = np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, d]])

    optimal_rotation_matrix = V @ coord_system_corrector @ U.T

    return optimal_rotation_matrix
