# -------------------------  Refs --------------------------------
# [Xgboost python API Manual](http://xgboost.readthedocs.io/en/latest/python/python_api.html)
# [Xgboost python usage introduction](http://xgboost.readthedocs.io/en/latest/python/python_intro.html)

import time
import cv2 as cv
import numpy as np
import xgboost as xgb

def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'error', float(sum(labels != (preds > 0.0))) / len(labels)

t = time.time()
TRAIN_NUM = "ALL"
TEST_NUM = "ALL"
dtrain = xgb.DMatrix("../features3d/feature3d_%s.txt"%TRAIN_NUM)
dtest = xgb.DMatrix("../features3d/feature3d_%s.txt"%TEST_NUM)

labels = dtrain.get_label()
print(labels)
# dtrain = xgb.DMatrix("./features3d/FLATTEN_DATASET.txt")
# dtest = xgb.DMatrix("./features3d/flatten_%s.txt"%TEST_NUM)

# param is a dictionary. you can refer to xgboost python intro for further info on its keys and available values. Its obj-func and eval can be defined by users
param = {'max_depth': 5, 'eta': 1, 'silent': 1}

watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 10

dst = xgb.train(param, dtrain, num_round, watchlist, logregobj, evalerror) # dst is xgb.Booster. 
dst.save_model("./hog3d.model")