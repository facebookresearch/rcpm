# Copyright (c) Facebook, Inc. and its affiliates.

import jax.numpy as np
from math import pi
import jax

def spherical_to_euclidean(theta_phi):
    single= theta_phi.ndim == 1
    if single:
        theta_phi = np.expand_dims(theta_phi, 0)
    theta, phi = np.split(theta_phi, 2, 1)
    return np.concatenate((
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ), 1)


def euclidean_to_spherical(xyz):
    single = xyz.ndim == 1
    if single:
        xyz = np.expand_dims(xyz, 0)
    x, y, z = np.split(xyz, 3, 1)
    return np.concatenate((
        np.arctan2(y, x),
        np.arccos(z)
    ), 1)

def S1euclideantospherical(euc_coords):
    return np.arctan2(euc_coords[:,1], euc_coords[:,0])

def productS1toTorus(theta1, theta2):
    R = 1
    r = 0.3

    x = (R + r * np.cos(theta1))*np.cos(theta2)
    y = (R + r * np.cos(theta1))*np.sin(theta2)
    z = r * np.sin(theta1)
    return x,y,z
