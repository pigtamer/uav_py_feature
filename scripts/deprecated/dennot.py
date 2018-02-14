# !/usr/bin/python

# this script is dedicated to extract st-cube or single-frame img data from annotations file of epfl uav video dataset for model training.
# all it's outputs must be tested to containing uav target as positive and non-target areas as nagative.
import re
import numpy as np
import cv2 as cv

# annot path = X:/uav/annot/drones/
def createDataset(video_file, annot_file, patch_size = 40, flight_object_type = 'drones' ):
    # annots: formated annot data. 
    # patch_size: (40, 40) tuple
    # sample_per_frame: 10
    # flight_object_type: 'drones' or 'planes'
    def fmt_annot(annot_file):
        # child func of createDataset.
        # format annotaion files by certain splitters.
        return 


    # --extract rois from annot data ---
    # -- return dataset_train