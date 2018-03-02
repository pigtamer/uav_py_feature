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
# ---------------------- PARAMS -------------------------------

CUBE_T, CUBE_Y, CUBE_X = (4, 64, 64)# define the size of each st-cube to be processed
HOG_SIZE = (int(np.ceil(CUBE_X / 4)), int(np.ceil(CUBE_T / 2)))
HOG_STEP = (int(np.ceil(CUBE_X / 4)), int(np.ceil(CUBE_T / 2)))
BCDIV = 3

GAU_SIGMA = (1, 3, 3) #(t,y,x)
IF_LOG = True

STEP = int(16)
# ---------------------------------------------------------------
Vex = []
fbuffer = deque()    # buffer for st-cube
time_stamp = 0 
tic = time.time()
while(True):
    ret, frame = cap.read()
    if not ret: break

    frame = frame[180: 360, 340: 540]

    # frame = frame[180: 480, 240: 540]
    
    
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # now is uint

    frame = im2double(frame)# caution: set each frame as double

    fbuffer.append(frame)

    # print(frame.shape)
    # T_GRID = np.arange(0, frame.shape[0], CUBE_T)   
    Y_GRID = np.arange(0, frame.shape[1] - (CUBE_X + STEP), STEP)
    X_GRID = np.arange(0, frame.shape[0] - (CUBE_Y + STEP), STEP)

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
                    patch = cv.resize(patch, (40, 40))
                    if IF_LOG: patch = cv.Laplacian(patch, cv.CV_64F)
                    stcube.append(patch)

                stcube = np.array(stcube)
                stcube = gauss3d.smooth3d(stcube, GAU_SIGMA)    
                fhog = myhog3d.compute(stcube, HOG_SIZE, HOG_STEP, BCDIV)
                # print(fhog.shape)
                HOG_GRID.append(fhog)

        HOG_GRID = np.array(HOG_GRID)
        DM_GRID = xgb.DMatrix(HOG_GRID)

        ranks = dst.predict(DM_GRID)
        print(np.max(ranks))
        plt.figure()
        plt.plot(ranks)
        plt.show()
        # ranks[ranks > 0.5 * np.max(ranks)] = 0
         
        idx = int(np.argmax(ranks))
        
        # for k in range(ranks.size):
        #     RANK_MAP[int(k%CUBE_Y), int(k / CUBE_Y)] = ranks[k]
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(RANK_MAP, cmap='gray', interpolation='nearest')
        # plt.subplot(122)
        # plt.imshow(PEAK_MAP, cmap='gray', interpolation='nearest')
        
        # plt.figure()
        # plt.imshow(frame)
        # plt.show()

        if np.max(ranks) > 0:
            v1 = ( int(STEP * (idx % int(len(X_GRID)))), int(STEP * int(int(idx) / int(len(X_GRID)))) )
            v2 = (v1[0] + CUBE_X, v1[1] + CUBE_Y)
            Vex.append([v1[1], v1[0]])
            print(time_stamp, (v1[1], v1[0]))

            cv.rectangle(frame, v1, v2, 0)
            cv.imshow("f", frame)
            cv.waitKey(24)
        else:
            print("<!> TARGET NOT FOUND <!>")
    time_stamp = time_stamp + 1
    if time_stamp > 100: break

Vex = np.array(Vex)
# print(Vex.shape)
plt.figure()
plt.scatter(Vex[:, 0], Vex[:, 1])
plt.show()
toc = time.time() - tic