# THIS IS PARSER FOR ANNOTATIONS. TEXT FILES WILL BE READ INTO NP-ARRAY
import re
import numpy as np

file_path = "X:/UAV/annot/drones/"
file_num = 18

# Separator for regular expressions
sepa_loc = r"\(((\d*),(.))*(\d*)\)"

# seap_data = 
file_annot = open(file_path + "Video_" + "%d"%file_num + '.txt', 'r')

sample_idx = 0; # init line number
locations = []
labels = []

for line in file_annot:
    sample_idx = sample_idx + 1;
    # print("ORIGINAL TEXT: " + line)
    searchObj = re.search(sepa_loc, line, re.M|re.I|re.S)
    if searchObj:
        tmpStr = searchObj.group()
        tmpStr = tmpStr[1:-1] # get rid of "()", the brackets

        tmpList = tmpStr.split(",")

        for k in range(len(tmpList)):
            tmpList[k] = int(tmpList[k])
        tmpList = tuple(tmpList) # so it wont be manipulated later

        print("Time:%d\t\t"%sample_idx, " Loc: ", tmpList)
        locations.append(tmpList)
        labels.append(1)
    else:
        print("Time:%d\t"%sample_idx, " Loc: ", "--- NO ENTRY ---")
        locations.append((-1, -1, -1, -1))# no target
        labels.append(0)
        
locations = np.array(locations) # convert list to np.array for later manip
print(locations) # optional
print(labels)