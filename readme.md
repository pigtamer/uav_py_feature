# References

## HoG

1. This is an [Opencv Example](https://raw.githubusercontent.com/sturkmen72/opencv/403682ff600098cebfdd1e3a4f51313342f1ad15/samples/cpp/train_HOG.cpp) with the func to visualize hog structure.

2. [CLI Example](https://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/)

3. A [Opencv HoG trainer on GitHub](https://github.com/DaHoC/trainHOG)

4. [Parameters Example on StackOverflow](https://stackoverflow.com/questions/27343614/opencv-hogdescriptor-compute-error). Caution: we **MUST** adjust it according to our patch size and other shits.

5. [HoG theory basis on learnopencv.com](https://www.learnopencv.com/histogram-of-oriented-gradients/)

6. [Forum](http://answers.opencv.org/question/10374/how-to-training-hog-and-use-my-hogdescriptor/)


## Regex

1. Python module "re" [Documentation](https://docs.python.org/3/library/re.html)

2. Remember separators for EPFL UAV dataset:
```python
    sepa_loc = r"\(((\d*),(.))*(\d*)\)"
    ...
    searchObj = re.search(sepa_loc, line, re.M|re.I|re.S)
```

## XGBoost

**[Git Deom Repo for Py](https://github.com/dmlc/xgboost/tree/master/demo/guide-python)**

1. [Example for custom obj func](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py)

2. [Xgboost python API Manual](http://xgboost.readthedocs.io/en/latest/python/python_api.html)

3. [Xgboost python usage introduction](http://xgboost.readthedocs.io/en/latest/python/python_intro.html)

## Trifles on python


## Other shits

Can try scikit module: sklearn.ensemble.GradientBoostingClassifier() as an alternative.