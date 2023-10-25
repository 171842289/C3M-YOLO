# C3M-YOLO
## Module
You could find the Lightweight Module C3M and Attention Module in the file named ``common.py``.

## Training
You can use the following command to start training in the command line, the data for your own datasets and this weights for pretrained one：  

``python train.py --data data.yaml --epochs 300 --weights '' --cfg yolov5s.yaml  --batch-size 16``

## Val
For testing your model, and you also could set relevant parameters, for example the weights for the model which you has trained:  

``python val.py``

