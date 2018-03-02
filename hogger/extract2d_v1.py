# !/usr/bin/python
# this is for extracting 2-d hog feature from video seq
import os
import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# sys.path.append("X:/UAV/py/fuckaround/try_cv_feature/hogger/")

import annot_parser # this is not wrong as we've added


# -----------------------------------------------------------------------------
# --- How to set these crap params?
# set parameters for opencv HOGDescriptor
# these paras should be manipulated according to size of our im patch

nbins = 9
winSize = (40, 40) # CAUTION: blocksize just runs on winsize range.

blockSize = (10, 10); blockStride = (5, 5); cellSize = (5, 5) # 1764 = 36 * 49 feature points

# blockSize = (20, 20); blockStride = (5, 5); cellSize = (10, 10) # 900 = 36 *25 feature points

# blockSize = (40, 40); blockStride = (20, 20); cellSize = (20, 20) # 36 = 36 * 1 feature points

# assistant params
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 0.1
gammaCorrection = True
nlevels = 64

# generate hog descriptor
hogger2d = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
# -----------------------------------------------------------------------------


data_path = "D:/Proj/UAV/dataset/drones/"
data_postfix = ".avi"
if not os.path.exists("../features"): os.makedirs("../features")

# parse annot into array
drones_nums = [1, 11, 12, 18, 19, 29, 37, 46, 47, 48, 49, 53, 55, 56]
TRAIN_SET_RANGE = [1] # select some videos by slicing

IF_SHOW_PATCH = False # warning: it can critically slow down extraction process
IF_PLOT_HOG_FEATURE = False

NEGA_SPF = 10
# parse videos in training set
t = time.time()
for VID_NUM in TRAIN_SET_RANGE: #---- do all those shits down here

    locations, labels = annot_parser.parse("X:/UAV/annot/drones/", VID_NUM)
    data_num = VID_NUM

    cap = cv.VideoCapture(data_path + "Video_%s"%data_num + data_postfix)
    # cap = cv.VideoCapture(0)
    file_out = open("../features/feature_%d.txt"%VID_NUM, 'w')

    # parse each video    
    time_stamp = 0
    while(True):
        ret, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        x_0 = locations[time_stamp][0] # 1
        x_1 = locations[time_stamp][2] # 3
        y_0 = locations[time_stamp][1] # 2
        y_1 = locations[time_stamp][3] # 4

        if not x_0 == -1 : 
            patch = frame[x_0:x_1, y_0:y_1]
            patch = cv.resize(patch, (40,40)) # size of target area varies in time so we resize each patch to a certain size, fitting HoG Descriptor.
        else:
            rand_nega_x = int(np.floor((frame.shape[0] - 40) * np.random.rand()))
            rand_nega_y = int(np.floor((frame.shape[1] - 40) * np.random.rand()))
            patch = frame[rand_nega_x : rand_nega_x + 40, rand_nega_y : rand_nega_y + 40]

        patch_hog = hogger2d.compute(patch)
        file_out.write("%d " % (labels[time_stamp]))
        for idx in range(patch_hog.size):
            # idx + 1 to fit libsvm format (xgb)
            file_out.write("%d:%f " % (idx, patch_hog[idx]))
        file_out.write('\n')


        if IF_PLOT_HOG_FEATURE: plt.plot(patch_hog); plt.show()
        if IF_SHOW_PATCH:
            cv.imshow("patch", patch) # if no target in annotated, then a negative sample is added from a random area on each frame. image window can flicker when negative samples are displayed.
            cv.waitKey(24)

        for k in range(NEGA_SPF):
            xn_0 = int(np.floor((frame.shape[0] - 40) * np.random.rand()))
            yn_0 = int(np.floor((frame.shape[1] - 40) * np.random.rand()))
            if ((xn_0 > x_0 - 40 and xn_0 < x_0 + 40) and (yn_0 > y_0 - 40 and yn_0 < y_0 + 40)): # avoid overlapping
                k = k - 1
                continue
            
            npatch = frame[xn_0 : xn_0 + 40, yn_0 : yn_0 + 40]
            n_patch_hog = hogger2d.compute(npatch)
            file_out.write("0 ")
            for idx in range(n_patch_hog.size):
                # idx + 1 to fit libsvm format (xgb)
                file_out.write("%d:%f " % (idx, n_patch_hog[idx]))
            file_out.write('\n')


            
        time_stamp = time_stamp + 1
        if time_stamp == locations.shape[0] : break

elapsed_time = time.time() - t
print("Dataset generated in %s sec"%elapsed_time)