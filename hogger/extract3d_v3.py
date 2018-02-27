# !/usr/bin/python

# Based on previous versions, this script extract many negative patches for each timestamp
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

TRAIN_SET_RANGE = grp_0
# TRAIN_SET_RANGE = [29]

# ---------------------- PARAMS --------------------------------
TRAIN_MODE = "strict"

SAVE_FEATURE = True
SAVE_EXTRA_NEGATIVE = True and SAVE_FEATURE

IF_SHOW_PATCH = not SAVE_FEATURE 
IF_PLOT_HOG_FEATURE = not SAVE_FEATURE

CUBE_T, CUBE_Y, CUBE_X = (4, 40, 40)# define the size of each st-cube to be processed
HOG_SIZE = (int(CUBE_X / 2), int(CUBE_T))
HOG_STEP = (int(CUBE_X / 2), int(CUBE_T))
BCDIV = 3

GAU_SIGMA = (1, 3, 3) #(t,y,x)
IF_LOG = True


NEGA_SPF = 10
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
    
    # buffer = deque()    # buffer for st-cube
    # n_buffer = deque()
    fbuffer = deque()    # buffer for st-cube
    
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

        
        # if x_0 == -1 : continue
        fbuffer.append(frame)

        # ----------------- ST-CUBE generation with deque buffer --------------|
        if len(fbuffer) == CUBE_T + 1 :
            fbuffer.popleft() # pop a frame from head when buffer is filled

            stcube = []
            

            xn_0 = int(np.floor((frame.shape[0] - (x_1 - x_0)) * np.random.rand()))
            yn_0 = int(np.floor((frame.shape[1] - (y_1 - y_0)) * np.random.rand()))
            for frms in fbuffer:
                rand_patch = frms[xn_0 : xn_0 + CUBE_X, yn_0 : yn_0 + CUBE_Y]
                rand_patch = cv.resize(rand_patch, (CUBE_X, CUBE_Y))
                if IF_LOG: rand_patch = cv.Laplacian(rand_patch, cv.CV_64F)
                
                # n_stcube.append(n_patch)
                if x_0 == -1:
                    stcube.append(rand_patch)
                else:
                    patch = cv.resize(frms[x_0:x_1, y_0:y_1], (CUBE_X, CUBE_Y))
                    if IF_LOG: patch = cv.Laplacian(patch, cv.CV_64F)
                    stcube.append(patch)

# -----------------------------positive --------------------------------------
            stcube = np.array(stcube)
            stcube = gauss3d.smooth3d(stcube, GAU_SIGMA)    
            FHOG3D = myhog3d.compute(stcube, HOG_SIZE, HOG_STEP, BCDIV)

            label_cube = labels[time_stamp - CUBE_T + 1: time_stamp + 1]
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
                plt.plot(FHOG3D)
                plt.title("[%s], VID[%d][%d / %d], s.t.[%s], fpc[%d], gauss[%s]"%(FINAL_LABEL_FOR_CUBE, VID_NUM, time_stamp, locations.shape[0], TRAIN_MODE, CUBE_T, GAU_SIGMA)  )              


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

            if CUBE_T < 5 and IF_SHOW_PATCH:
                plt.figure(figsize = (4* CUBE_T, 6))
                for k in range(CUBE_T):
                    plt.subplot(1, CUBE_T, k + 1)
                    plt.title(label_cube[k])
                    plt.imshow(stcube[k, :, :])
                plt.show()



# ---------------------------------Negative------------------------------------
            l_n_stcube = [] # list of negas
            
            for idx in range(NEGA_SPF):
                n_stcube = []

                xn_0 = int(np.floor((frame.shape[0] - (x_1 - x_0)) * np.random.rand()))
                yn_0 = int(np.floor((frame.shape[1] - (y_1 - y_0)) * np.random.rand()))
                for frms in fbuffer:
                    n_patch = frms[xn_0 : xn_0 + CUBE_X, yn_0 : yn_0 + CUBE_Y]
                    n_patch = cv.resize(n_patch, (CUBE_X, CUBE_Y))
                    if IF_LOG: n_patch = cv.Laplacian(n_patch, cv.CV_64F)
                    n_stcube.append(n_patch)
                n_stcube = np.array(n_stcube)
                n_stcube = gauss3d.smooth3d(n_stcube, GAU_SIGMA)
                l_n_stcube.append(n_stcube)

            k = 0
            for n_stcube in l_n_stcube:
                N_FHOG3D = myhog3d.compute(n_stcube,  HOG_SIZE, HOG_STEP, BCDIV)

                if SAVE_EXTRA_NEGATIVE:
                    file_out.write("%d "%0)
                    for idx in range(N_FHOG3D.size):
                        # idx + 1 to fit libsvm format (xgb)
                        file_out.write("%d:%f " % (idx + 1, N_FHOG3D[idx]))
                    file_out.write('\n')

                    group_file_out.write("%d "%0)
                    for idx in range(N_FHOG3D.size):
                        # idx + 1 to fit libsvm format (xgb)
                        group_file_out.write("%d:%f " % (idx + 1, N_FHOG3D[idx]))
                    group_file_out.write('\n')

                if IF_PLOT_HOG_FEATURE and k == 0:
                    plt.plot(N_FHOG3D)
                    plt.title("Random Negative Sample")
                    plt.show()
                k  = k + 1

            
            if CUBE_T < 5 and IF_SHOW_PATCH:
                plt.figure(figsize = (4* CUBE_T, 6))                
                for k in range(CUBE_T):
                    plt.subplot(1, CUBE_T, k + 1)
                    plt.imshow(n_stcube[k, :, :])
                plt.show()


                

        time_stamp = time_stamp + 1
        if time_stamp == locations.shape[0] : break

    toc = time.time() - tic
    print("HoG Feature Extracted in : %5.3f sec;"%toc)
    # if len(buffer) == CUBE_T: print("Buffer size correct: %d for %d."%(len(buffer), CUBE_T))

TOC = time.time() - TIC
print("/ / / / / / / / / / / /\nDataset generated in: %5.3f sec."%TOC)
