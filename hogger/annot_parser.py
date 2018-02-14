# THIS IS PARSER FOR ANNOTATIONS. TEXT FILES WILL BE READ INTO NP-ARRAY
import re # for regular expressions
import numpy as np

# may not be needed

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


def parse(file_path = "X:/UAV/annot/drones/", file_num = 1, sepa_loc = r"\(((\d*),(.))*(\d*)\)", IF_PRINT = False):
    # parsinf function.
    file_annot = open(file_path + "Video_" + "%d"%file_num + '.txt', 'r')
    locations = []
    labels = []

    sample_idx = 0 # line num counter for input file
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
            # tmpList = tuple(tmpList) # so it wont be manipulated later

            if IF_PRINT: print("Time:%d\t\t"%sample_idx, " Loc: ", tmpList)
            locations.append(tmpList)
            labels.append(1)
        else:
            if IF_PRINT: print("Time:%d\t"%sample_idx, " Loc: ", "--- NO ENTRY ---")
            locations.append((-1, -1, -1, -1))# no target
            labels.append(0)
            
    locations = np.array(locations) # convert list to np.array for later manip
    if IF_PRINT:
        print(locations, "\n", labels) # optional
        # print(labels)
    print("Parsing file <%s> complete."%(file_path + "Video_" + "%d"%file_num + '.txt'))
    return locations, labels


# ----------------------------------------------------------
#%% TEST PARSER MODULE

# file_num = 18
# file_path = "X:/UAV/annot/drones/"
# sepa_loc = r"\(((\d*),(.))*(\d*)\)"
# IF_PRINT = False

# loc, lab = parse(file_path, file_num, sepa_loc, IF_PRINT)
# # print(loc, "\n", lab)

