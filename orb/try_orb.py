import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("X:/cvImg/koutaku.jpg", 0)

cv.imshow("sample", img); cv.waitKey(0)

# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

cv.imshow("orb features", img2); cv.waitKey(0)