echo off 
python ./hogger/extract3d_v1.py
python ./ml/xgb_3d_train.py

REM python ./ml/xgb_3d_run.py
