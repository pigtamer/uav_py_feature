# !/usr/bin/python
import os
import time
import cv2 as cv
import numpy as np
import annot_parser
import myhog3d
import matplotlib.pyplot as plt
# import seaborn as sns
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
# TRAIN_SET_RANGE = drones_nums[0:1] # select some videos

TRAIN_SET_RANGE = np.array([18])
IF_SHOW_PATCH = False # warning: it can critically slow down extraction process
IF_PLOT_HOG_FEATURE = False

# parse videos in training set
# VID_NUM = 1; # for single test
for VID_NUM in TRAIN_SET_RANGE: #---- do all those shits down here
    #   {
    tic = time.time()
    
    locations, labels = annot_parser.parse("X:/UAV/annot/drones/", VID_NUM)
    data_num = VID_NUM

    cap = cv.VideoCapture(data_path + "Video_%s"%data_num + data_postfix)
    file_out = open("../features3d/feature3d_%d.txt"%VID_NUM, 'w')

    # parse each video    
    time_stamp = 0
    CUBE_X, CUBE_Y, CUBE_T = 40 , 40, 4; 

    buffer = deque()    # buffer 
    while(True):
        ret, frame = cap.read()
        if not ret: break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # now is uint

        frame = im2double(frame)# caution: each frame as double
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

        if len(buffer) == CUBE_T + 1: 
            buffer.popleft() # pop a frame from head when buffer is filled
            stcube = np.array(buffer)
            # print(stcube.shape)

            if CUBE_T < 5 and IF_SHOW_PATCH:
                for k in range(CUBE_T):
                    plt.subplot(1, CUBE_T, k + 1)
                    plt.imshow(stcube[:][:][k])
                plt.show()

            FHOG3D = myhog3d.hog3d(stcube, labels[time_stamp], (10, 4), (10, 4), 2, False, False)

            # if IF_PLOT_HOG_FEATURE:
            #     plt.plot(FHOG3D);plt.title(labels[time_stamp]);plt.show()

            file_out.write("%d " % (labels[time_stamp]))
            for idx in range(FHOG3D.size):
                # idx + 1 to fit libsvm format (xgb)
                file_out.write("%d:%f " % (idx + 1, FHOG3D[idx]))
            file_out.write('\n')

            time_stamp = time_stamp + 1
            if time_stamp == locations.shape[0] : break

    toc = time.time() - tic
    print("Time elapsed: %s", toc)
    # if len(buffer) == CUBE_T: print("Buffer size correct: %d for %d."%(len(buffer), CUBE_T))