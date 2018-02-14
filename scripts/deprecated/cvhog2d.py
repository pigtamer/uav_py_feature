# !/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv

sns.set()

def shiftWin(img, idxRange):
    # -- img as numpy mat
    # this func shifts window on image if necessary
    imPatch = img[idxRange[0]: idxRange[2], idxRange[1]: idxRange[3]]
    return imPatch


def showIm(WINDOW_NAME, im):
    # this func shows image with opencv routine.
    # may be replaced by plt.imshow(...)
    cv.imshow(WINDOW_NAME, im)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def dumpDataForXGB(array_of_np_array, label_of_each_arr, filename = "feature"):
    # array reshaped as 1*N, in libsvm format
    # length of label should be the same as img array
    # rows of img arr, smaples of imgs
    num_img_samples = array_of_np_array.shape[0]
    with open(filename, 'w') as file_out:
        for k in range(num_img_samples):
            file_out.write("%d " % (label_of_each_arr[k]))
            for idx in range(array_of_np_array[k].size):
                # idx + 1 to fit libsvm format (xgb)
                file_out.write("%d:%f " % (idx + 1, array_of_np_array[k][idx]))
            file_out.write('\n')
    return


def splitForXrossVal(prop = 0.1, dump_filename = "feature"):
    # this func splits dumped data file into test and train set
    # when original file canot fit our needs.
    # prop: proportion of test set in whole dataset

    return

def directXrossVal(dump_filename):
    # directly cross validate
    pass
    return 

def myHoG2dInit():
    # reserved for general hog descriptor init and compute.
    return 

TEST_IM_PATH = "X:/cvImg/"  # -- im for simple test
# -- real vid seq for advanced test.
VIDEO_PATH = "D:/Proj/UAV/dataset/drones/"


im = cv.imread(TEST_IM_PATH + "koutaku.jpg")
im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
im = im[160:160 + 80, 140:140 + 80]
showIm("im", im)

# --- How to set these crap params?
# set parameters for opencv HOGDescriptor
# these paras should be manipulated according to size of our im patch

nbins = 9
winSize = (80, 80) # CAUTION: blocksize just runs on winsize range.

blockSize = (40, 40)
blockStride = (20, 20)
cellSize = (20, 20) # 324 = 36 * 9 feature points

# assistant params
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 0.1
gammaCorrection = True
nlevels = 64

# generate hog descriptor
hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

# set parameters for opencv hog compute
# compute(img[, winStride[, padding[, locations]]]) -> descriptor

# these parameters will define how the window is slided on img.
# I leave them as default.
winStride = (4, 4)
padding = (8, 8)
locations = ((10, 20),)

# hist = hog.compute(im, winStride, padding, locations)
hist = hog.compute(im)

plt.plot(hist) # Vis feature series.
plt.show()

hist = np.transpose(hist)

print("%d feature points generated."%hist.size)
dumpDataForXGB(hist, 0.5 + np.random.rand(hist.size))
