import time
import cv2 as cv
import numpy as np
import xgboost as xgb

# -----------------------------------------------------------------------------
nbins = 9
winSize = (40, 40) # CAUTION: blocksize just runs on winsize range.

blockSize = (10, 10); blockStride = (5, 5); cellSize = (5, 5) # 1764 = 36 * 49 feature points
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 0.1
gammaCorrection = True
nlevels = 64

# generate hog descriptor
hogger2d = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
# -----------------------------------------------------------------------------
dst =xgb.Booster()
dst.load_model("./hog2d.model")
#%%
data_path = "D:/Proj/UAV/dataset/drones/"
data_postfix = ".avi"
data_num = 12
cap = cv.VideoCapture(data_path + "Video_%s"%data_num + data_postfix)

STEP = 10
CSIZE = 64
time_stamp = 0
while(True):
    ret, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    Y_GRID = np.arange(0, frame.shape[1] - (40 + STEP), STEP)
    X_GRID = np.arange(0, frame.shape[0] - (40 + STEP), STEP)

    ech_frame = [] # container for pathces on each frame
    for x_0 in X_GRID:
        x_1 = x_0 + CSIZE
        for y_0 in Y_GRID:
            y_1 = y_0 + CSIZE
            patch = frame[x_0:x_1, y_0:y_1]   
            patch = cv.resize(patch, (40, 40))     
            patchhog = hogger2d.compute(patch)
            ech_frame.append(patchhog)
    ech_frame = np.array(ech_frame)
    ech_frame = ech_frame[:, :, 0]
    
    evDM = xgb.DMatrix(ech_frame)

    ranks = dst.predict(evDM)
    idx = np.argmax(ranks)

    # print(idx)
    # print("(%d, %d)"%(idx / (X_PATCH_NUM_MAX + 1), idx % (X_PATCH_NUM_MAX + 1)))
    v1 = ( int(STEP * (idx % int(len(X_GRID)))), int(STEP * int(int(idx) / int(len(X_GRID)))) )
    v2 = (v1[0] + CSIZE, v1[1] + CSIZE)
    
    cv.rectangle(frame, v1, v2, 0)
    cv.imshow("f", frame)
    cv.waitKey(240)

    time_stamp = time_stamp + 1
    