import numpy as np
import cv2 as cv

from scipy.ndimage import filters

def smooth3d(cuboid, sigma):
    # the cuboid input should can be multidimensional, and its dims should correspond to the length of sigma
    assert len(cuboid.shape) == len(sigma)
    DIMS = len(cuboid.shape)
    smoothed = cuboid
    for axis in range(DIMS):
        smoothed =  filters.gaussian_filter1d(smoothed, sigma[axis], axis)
    return smoothed