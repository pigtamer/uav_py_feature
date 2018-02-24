# !/usr/bin/python
import os
import time
import cv2 as cv
import numpy as np
import annot_parser
import matplotlib.pyplot as plt
from collections import deque
# import sklearn.??? as ??? # -- if needed

## ---- REF ---
# [1] [A Spatio-Temporal Descriptor Based on 3D-Gradients](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwiirOSqjqPZAhUpwVQKHUpyB3AQFggsMAA&url=https%3A%2F%2Fhal.inria.fr%2Finria-00514853%2Fdocument&usg=AOvVaw0mijsjePgJYJ4jAGXSxANF)
# [2] [Behavior recognition via sparse spatio-temporal features](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=0ahUKEwicrKfEjqPZAhVFylQKHRjaB3AQFgg6MAE&url=https%3A%2F%2Fpdollar.github.io%2Ffiles%2Fpapers%2FDollarVSPETS05cuboids.pdf&usg=AOvVaw3P5KcCPAyHlxoHcp0dg-Xr)


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


data_path = "D:/Proj/UAV/dataset/drones/"
data_postfix = ".avi"
if not os.path.exists("../features3d"): os.makedirs("../features3d")

cap = cv.VideoCapture()

drones_nums = [1, 11, 12, 18, 19, 29, 37, 46, 47, 48, 49, 53, 55, 56]
TRAIN_SET_RANGE = drones_nums[0:1] # select some videos

IF_SHOW_PATCH = True # warning: it can critically slow down extraction process
IF_PLOT_HOG_FEATURE = False

# parse videos in training set
VID_NUM = 1; # for single test
t = time.time()
# for VID_NUM in TRAIN_SET_RANGE: #---- do all those shits down here
#   {
locations, labels = annot_parser.parse("X:/UAV/annot/drones/", VID_NUM)
data_num = VID_NUM

cap = cv.VideoCapture(data_path + "Video_%s"%data_num + data_postfix)
file_out = open("../features3d/feature3d_%d.txt"%VID_NUM, 'w')

# parse each video    
time_stamp = 0
CUBE_X, CUBE_Y, CUBE_T = 40 , 40, 4; assert CUBE_T > 1 and CUBE_T < 50

buffer = deque()    # buffer 
while(True):
    ret, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # now is uint

    frame = im2double(frame)
    x_0 = locations[time_stamp][0] # 1
    x_1 = locations[time_stamp][2] # 3
    y_0 = locations[time_stamp][1] # 2
    y_1 = locations[time_stamp][3] # 4

    if not x_0 == -1 : 
        patch = frame[x_0:x_1, y_0:y_1]
        patch = cv.resize(patch, (CUBE_X, CUBE_Y)) # size of target area varies in time so we resize each patch to a certain size, fitting HoG Descriptor.
    else:
        rand_nega_x = int(np.floor((frame.shape[0] - CUBE_X) * np.random.rand()))
        rand_nega_y = int(np.floor((frame.shape[1] - CUBE_Y) * np.random.rand()))
        patch = frame[rand_nega_x : rand_nega_x + CUBE_X, rand_nega_y : rand_nega_y + CUBE_Y]

    # ----------------- ST-CUBE generation with deque buffer --------------|
    buffer.append(patch) # push a patch to the rear of stcube    
    if len(buffer) == CUBE_T: 
        buffer.popleft() # pop a frame from head when buffer is filled
    stcube = np.array(buffer)
    # ---------------------------------------------------------------------||


    # CALC HOG3d IN EACH ST_CUBE
    #{
    if time_stamp >= CUBE_T:
        [dx, dy, dt] = np.gradient(stcube, 1, 1, 1) # NAIVE grad in 3-dims
        for k in range(CUBE_T):
            plt.subplot(1, CUBE_T, k + 1)
            plt.imshow(stcube[:][:][k])
        plt.show()

    # }

    # -----------------------------------------------------------------------
    # # if IF_PLOT_HOG_FEATURE: plt.plot(patch_hog); plt.show()
    # if IF_SHOW_PATCH:
    #     cv.imshow("patch", patch) # show patch in the latest frame of a stcube. if no target in annotated, then a negative sample is added from a random area on each frame. image window can flicker when negative samples are displayed.
    #     # cv.waitKey(0) # test
    #     cv.waitKey(24) # run

    # file_out.write("%d " % (labels[time_stamp]))
    # for idx in range(patch_hog.size):
    #     # idx + 1 to fit libsvm format (xgb)
    #     file_out.write("%d:%f " % (idx + 1, patch_hog[idx]))
    # file_out.write('\n')

    time_stamp = time_stamp + 1;
    if time_stamp == locations.shape[0] : break


# } END LOOP

elapsed_time = time.time() - t
if len(buffer) == CUBE_T: print("Buffer size correct: %d for %d."%(len(buffer), CUBE_T))

print("Dataset generated in %s sec"%elapsed_time)