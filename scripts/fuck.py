import cv2 as cv
import numpy as np

a = cv.imread("X:/cvImg/cross.jpg")

[dx,dy,dt] = np.gradient(a, 1, 1, 1)

cv.imshow("res", dx)
cv.waitKey(0)