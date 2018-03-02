echo off 
python ./hogger/extract2d_v1.py
python ./ml/xgb_2d_train.py

python ./ml/xgb_2d_run.py
