import time
import cv2 as cv
import numpy as np
import xgboost as xgb
from collections import deque
import matplotlib.pyplot as plt
import myhog3d
import gauss3d

# --------------------- CUSTOM FUNC -------------------------
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    if max_val != min_val:
        out = (im.astype('float') - min_val) / (max_val - min_val)
    else:
        out = im.astype('float') / 255
    return out

## -------------------- LOAD MODEL ----------------------------
dst =xgb.Booster()
dst.load_model("./hog3d.model")
## -------------------- DATASET -------------------------------
data_path = "D:/Proj/UAV/dataset/drones/"
data_postfix = ".avi"
data_num = 1
cap = cv.VideoCapture(data_path + "Video_%s"%data_num + data_postfix)
# ---------------------- PARAMS --------------------------------
TRAIN_MODE = "strict"

CUBE_T, CUBE_Y, CUBE_X = (4, 64, 64)# define the size of each st-cube to be processed
CSTEP, TSTEP = (CUBE_X, CUBE_T)
HOG_SIZE = (int(CUBE_X / 4), int(CUBE_T / 2))
HOG_STEP = (int(CUBE_X / 4), int(CUBE_T / 2))
BCDIV = 3

GAU_SIGMA = (1, 1, 1) #(t,y,x)
IF_LOG = False

STEP = CSTEP
# ---------------------------------------------------------------

fbuffer = deque()    # buffer for st-cube
time_stamp = 0    
tic = time.time()
while(True):
    ret, frame = cap.read()
    if not ret: break
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # now is uint

    frame = im2double(frame)# caution: set each frame as double

    fbuffer.append(frame)

    print(frame.shape)
    # T_GRID = np.arange(0, frame.shape[0], CUBE_T)   
    Y_GRID = np.arange(0, frame.shape[1] - CUBE_Y, STEP)
    X_GRID = np.arange(0, frame.shape[0] - CUBE_X, STEP)

    if len(fbuffer) == CUBE_T + 1 :
        fbuffer.popleft() # pop a frame from head when buffer is filled

        HOG_GRID = []        
        for x_0 in X_GRID:
            x_1 = x_0 + CUBE_X
            for y_0 in Y_GRID:
                y_1 = y_0 + CUBE_Y
                stcube = []
                for frms in fbuffer:
                    patch = frms[x_0:x_1, y_0:y_1]
                    if IF_LOG: patch = cv.Laplacian(patch, cv.CV_64F)
                    stcube.append(patch)

                stcube = np.array(stcube)
                stcube = gauss3d.smooth3d(stcube, GAU_SIGMA)    
                fhog = myhog3d.compute(stcube, HOG_SIZE, HOG_STEP, BCDIV)

                HOG_GRID.append(fhog)
        HOG_GRID = np.array(HOG_GRID)
        print(HOG_GRID.shape)        
        # # HOG_GRID = HOG_GRID[:, :, 0]
        DM_GRID = xgb.DMatrix(HOG_GRID)

        ranks = dst.predict(DM_GRID)
        # ranks[ranks > 0.5 * np.max(ranks)] = 0
         
        idx = np.argmax(ranks)
        RANK_MAP = ranks.reshape((int(frame.shape[0]/CUBE_X) - 1, int(frame.shape[1]/CUBE_Y) - 1))

        PEAK_MAP = RANK_MAP.copy()
        PEAK_MAP[PEAK_MAP < np.max(PEAK_MAP)] = 0
        # RANK_MAP = np.zeros(((int(frame.shape[0]/CUBE_X), int(frame.shape[1]/CUBE_Y))))
        v1 = (int(CUBE_X * (idx % int(frame.shape[1] / CUBE_Y))), int(CUBE_Y * (idx / int(frame.shape[1] / CUBE_Y))))
        v2 = (v1[0] + CUBE_X, v1[1] + CUBE_Y)
        
        print(CUBE_X * (idx % int(frame.shape[1] / CUBE_Y)), CUBE_Y * (idx / int(frame.shape[1] / CUBE_Y)) )
        # for k in range(ranks.size):
        #     RANK_MAP[int(k%CUBE_Y), int(k / CUBE_Y)] = ranks[k]
        plt.figure()
        plt.subplot(121)
        plt.imshow(RANK_MAP, cmap='gray', interpolation='nearest')
        plt.subplot(122)
        plt.imshow(PEAK_MAP, cmap='gray', interpolation='nearest')
        
        plt.figure()
        plt.imshow(frame)
        plt.show()

        cv.rectangle(frame, v1, v2, 0)
        cv.imshow("f", frame)
        cv.waitKey(0)





    time_stamp = time_stamp + 1

toc = time.time() - tic