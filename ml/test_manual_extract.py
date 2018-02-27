import os
import time
import cv2 as cv
import numpy as np
import xgboost as xgb
from collections import deque
import matplotlib.pyplot as plt
import myhog3d
import gauss3d

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    if max_val != min_val:
        out = (im.astype('float') - min_val) / (max_val - min_val)
    else:
        out = im.astype('float') / 255
    return out

data_path = "D:/Proj/UAV/dataset/drones/"
data_postfix = ".avi"

cap = cv.VideoCapture()

## -------------------- DATASET -------------------------------
drones_nums = [1, 11, 12, 18, 19, 29, 37, 46, 47, 48, 49, 53, 55, 56]
grp_ALL = drones_nums
grp_0 = [1, 11, 12]
grp_1 = [18, 19, 29]
grp_2 = [37]
grp_3 = [46, 47, 48, 49]
grp_4 = [53, 55, 56]

TRAIN_SET_RANGE = grp_0
# TRAIN_SET_RANGE = [29]

# ---------------------- PARAMS --------------------------------
TRAIN_MODE = "strict"

SAVE_FEATURE = True
SAVE_EXTRA_NEGATIVE = True and SAVE_FEATURE

IF_SHOW_PATCH = not SAVE_FEATURE 
IF_PLOT_HOG_FEATURE = not SAVE_FEATURE

CUBE_T, CUBE_Y, CUBE_X = (4, 64, 64)# define the size of each st-cube to be processed
HOG_SIZE = (int(CUBE_X / 2), int(CUBE_T))
HOG_STEP = (int(CUBE_X / 2), int(CUBE_T))
BCDIV = 3

GAU_SIGMA = (1, 1, 1) #(t,y,x)
IF_LOG = False


NEGA_SPF = 10
