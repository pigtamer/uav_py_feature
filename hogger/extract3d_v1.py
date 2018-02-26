# !/usr/bin/python
import os
import time
import cv2 as cv
import numpy as np
# import seaborn as sns
import annot_parser
import myhog3d
import gauss3d
import matplotlib.pyplot as plt
# import seaborn as sns
from collections import deque

# sns.set()
# import sklearn.??? as ??? # -- if needed

## ---- REF ---
# [1] [A Spatio-Temporal Descriptor Based on 3D-Gradients](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwiirOSqjqPZAhUpwVQKHUpyB3AQFggsMAA&url=https%3A%2F%2Fhal.inria.fr%2Finria-00514853%2Fdocument&usg=AOvVaw0mijsjePgJYJ4jAGXSxANF)
# [2] [Behavior recognition via sparse spatio-temporal features](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=0ahUKEwicrKfEjqPZAhVFylQKHRjaB3AQFgg6MAE&url=https%3A%2F%2Fpdollar.github.io%2Ffiles%2Fpapers%2FDollarVSPETS05cuboids.pdf&usg=AOvVaw3P5KcCPAyHlxoHcp0dg-Xr)


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
if not os.path.exists("../features3d"): os.makedirs("../features3d")

cap = cv.VideoCapture()

## -------------------- DATASET -------------------------------
drones_nums = [1, 11, 12, 18, 19, 29, 37, 46, 47, 48, 49, 53, 55, 56]
grp_ALL = drones_nums
grp_0 = [1, 11, 12]
grp_1 = [18, 19, 29]
grp_2 = [37]
grp_3 = [46, 47, 48, 49]
grp_4 = [53, 55, 56]

TRAIN_SET_RANGE = grp_2
# TRAIN_SET_RANGE = [29]


# ---------------------- PARAMS --------------------------------
TRAIN_MODE = "strict"
SAVE_FEATURE = True
SAVE_EXTRA_NEGATIVE = True and SAVE_FEATURE

IF_SHOW_PATCH = not SAVE_FEATURE 
IF_PLOT_HOG_FEATURE = not SAVE_FEATURE

CUBE_T, CUBE_Y, CUBE_X = (4, 40, 40)# define the size of each st-cube to be processed
HOG_SIZE = (int(CUBE_X / 4), int(CUBE_T))
HOG_STEP = (int(CUBE_X / 4), int(CUBE_T))
BCDIV = 3
GAU_SIGMA = (1, 2, 2) #(t,y,x)

group_file_out = open("../features3d/feature3d_ALL.txt", 'w')
# NEGATIVE_SAMPLES_PER_FRAME = 1
# parse videos in training set
TIC = time.time()
for VID_NUM in TRAIN_SET_RANGE: #---- do all those shits down here
    #   {
    
    locations, labels = annot_parser.parse("X:/UAV/annot/drones/", VID_NUM)
    data_num = VID_NUM

    cap = cv.VideoCapture(data_path + "Video_%s"%data_num + data_postfix)
    file_out = open("../features3d/feature3d_%d.txt"%VID_NUM, 'w')

    # parse each video    

    tic = time.time()
    
    buffer = deque()    # buffer for st-cube
    n_buffer = deque()

    time_stamp = 0    
    while(True):
        ret, frame = cap.read()
        if not ret: break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # now is uint

        frame = im2double(frame)# caution: set each frame as double

        # the coord range of each st-cube
        x_0 = locations[time_stamp][0] # 1
        x_1 = locations[time_stamp][2] # 3
        y_0 = locations[time_stamp][1] # 2
        y_1 = locations[time_stamp][3] # 4

        if time_stamp % CUBE_T == 0:
            xn_0 = int(np.floor((frame.shape[0] - CUBE_X) * np.random.rand()))
            yn_0 = int(np.floor((frame.shape[1] - CUBE_Y) * np.random.rand()))

            # xn_0, yn_0 = 100, 100 # let negative cube stay at somewhere
            n_patch = frame[xn_0 : xn_0 + CUBE_X, yn_0 : yn_0 + CUBE_Y]
        
        if not x_0 == -1 : # annot-parser would return coord as -1 if no target is in current frame
            patch = frame[x_0:x_1, y_0:y_1]
            patch = cv.resize(patch, (CUBE_X, CUBE_Y)) # size of target area varies in time so we resize each patch to a certain size, fitting HoG Descriptor.
        else:
            patch = n_patch

        # ----------------- ST-CUBE generation with deque buffer --------------|
        buffer.append(patch) # push a patch to the rear of stcube    
        n_buffer.append(n_patch)

        if len(buffer) == CUBE_T + 1 and len(n_buffer) == CUBE_T + 1: 
            buffer.popleft() # pop a frame from head when buffer is filled
            stcube = np.array(buffer)

            n_buffer.popleft() # pop a frame from head when buffer is filled
            n_stcube = np.array(n_buffer)


            stcube = gauss3d.smooth3d(stcube, GAU_SIGMA)
            n_stcube = gauss3d.smooth3d(n_stcube, GAU_SIGMA)
            
            FHOG3D = myhog3d.compute(stcube, HOG_SIZE, HOG_STEP, BCDIV)
            N_FHOG3D = myhog3d.compute(n_stcube,  HOG_SIZE, HOG_STEP, BCDIV)


            label_cube = labels[time_stamp - CUBE_T + 1: time_stamp + 1]
            
            if CUBE_T < 5 and IF_SHOW_PATCH:
                plt.figure(figsize = (4* CUBE_T, 6))
                for k in range(CUBE_T):
                    plt.subplot(1, CUBE_T, k + 1)
                    plt.title(label_cube[k])
                    plt.imshow(stcube[k, :, :])
                plt.show()

                plt.figure(figsize = (4* CUBE_T, 6))                
                for k in range(CUBE_T):
                    plt.subplot(1, CUBE_T, k + 1)
                    plt.imshow(n_stcube[k, :, :])
                plt.show()

            if TRAIN_MODE == "strict":
                FINAL_LABEL_FOR_CUBE = 1
                for label_of_frame in label_cube:
                    FINAL_LABEL_FOR_CUBE  = FINAL_LABEL_FOR_CUBE and label_of_frame
            elif TRAIN_MODE == "loose":
                FINAL_LABEL_FOR_CUBE = 0
                for label_of_frame in label_cube:
                    FINAL_LABEL_FOR_CUBE  = FINAL_LABEL_FOR_CUBE or label_of_frame
            elif TRAIN_MODE == "current":
                FINAL_LABEL_FOR_CUBE = labels[time_stamp]
            else:
                FINAL_LABEL_FOR_CUBE = labels[time_stamp]


            if IF_PLOT_HOG_FEATURE:
                plt.figure(figsize=(8, 12))
                plt.subplot(2, 1, 1)
                plt.plot(FHOG3D)
                plt.title("[%s], VID[%d][%d / %d], s.t.[%s], fpc[%d], gauss[%s]"%(FINAL_LABEL_FOR_CUBE, VID_NUM, time_stamp, locations.shape[0], TRAIN_MODE, CUBE_T, GAU_SIGMA)  )              
                plt.subplot(2, 1, 2)
                fig_2 = plt.plot(N_FHOG3D)
                plt.title("Random Negative Sample")
                plt.show()

            # assert label_cube[-1] == labels[time_stamp]
            if SAVE_FEATURE:
                file_out.write("%d " % (FINAL_LABEL_FOR_CUBE))
                for idx in range(FHOG3D.size):
                    # idx + 1 to fit libsvm format (xgb)
                    file_out.write("%d:%f " % (idx + 1, FHOG3D[idx]))
                file_out.write('\n')

                group_file_out.write("%d " % (FINAL_LABEL_FOR_CUBE))
                for idx in range(FHOG3D.size):
                    # idx + 1 to fit libsvm format (xgb)
                    group_file_out.write("%d:%f " % (idx + 1, FHOG3D[idx]))
                group_file_out.write('\n')

            if SAVE_EXTRA_NEGATIVE:
                file_out.write("%d "%0)
                for idx in range(FHOG3D.size):
                    # idx + 1 to fit libsvm format (xgb)
                    file_out.write("%d:%f " % (idx + 1, N_FHOG3D[idx]))
                file_out.write('\n')

                group_file_out.write("%d "%0)
                for idx in range(FHOG3D.size):
                    # idx + 1 to fit libsvm format (xgb)
                    group_file_out.write("%d:%f " % (idx + 1, N_FHOG3D[idx]))
                group_file_out.write('\n')
                

        time_stamp = time_stamp + 1
        if time_stamp == locations.shape[0] : break

    toc = time.time() - tic
    print("HoG Feature Extracted in : %5.3f sec;"%toc)
    # if len(buffer) == CUBE_T: print("Buffer size correct: %d for %d."%(len(buffer), CUBE_T))

TOC = time.time() - TIC
print("/ / / / / / / / / / / /\nDataset generated in: %5.3f sec."%TOC)
