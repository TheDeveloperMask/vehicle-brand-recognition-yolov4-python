# Vehicle Make Recognition using YOLOv4 Object Detector

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Python example for using [Spectrico's car brand classifier](http://spectrico.com/car-make-model-recognition.html). It consists of an object detector for finding the cars, and a classifier to recognize the brands of the detected cars. The object detector is an implementation of YOLOv4 (OpenCV DNN backend). YOLOv4 weights were downloaded from [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights). The classifier is based on MobileNet v3 (Alibaba MNN backend).

---
## Object Detection and Classification in images
This example takes an image as input, detects the cars using YOLOv4 object detector, crops the car images, resizes them to the input size of the classifier, and recognizes the brand of each car. The result is shown on the display and saved as output.jpg image file.


#### Usage
Use --help to see usage of car_color_classifier_yolo4.py:
```
$ python car_color_classifier_yolo4.py --image cars.jpg
```
```
$ python car_color_classifier_yolo4.py [-h] [--yolo MODEL_PATH] [--confidence CONFIDENCE] [--threshold THRESHOLD] [--image]

required arguments:
  -i, --image              path to input image

optional arguments:
  -h, --help               show this help message and exit
  -y, --yolo MODEL_PATH    path to YOLO model weight file, default yolo-coco
  --confidence CONFIDENCE  minimum probability to filter weak detections, default 0.5
  --threshold THRESHOLD    threshold when applying non-maxima suppression, default 0.3
```
![image](https://github.com/spectrico/vehicle-brand-recognition-yolov4-python/blob/master/output.jpg?raw=true)

---
## Dependencies
  pip install numpy
  
  pip install opencv-python
  
  pip install MNN
  
  If you use Windows, the OpenCV have to be installed from: https://www.lfd.uci.edu/~gohlke/pythonlibs/

  yolov4.weights must be downloaded from [https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) and saved in folder yolov4

---
## Credits
The examples are based on the tutorial by Adrian Rosebrock: [YOLO object detection with OpenCV](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)

The YOLOv4 object detector is from: [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
```
@article{bochkovskiy2020yolov4,
  title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
  author={Bochkovskiy, Alexey and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2004.10934},
  year={2020}
}
```
The car brand classifier is based on MobileNetV3 mobile architecture: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
```
@inproceedings{howard2019searching,
  title={Searching for mobilenetv3},
  author={Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and others},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={1314--1324},
  year={2019}
}
```

The runtime library of the classifier is [MNN](https://github.com/alibaba/MNN)
```
@inproceedings{alibaba2020mnn,
  author = {Jiang, Xiaotang and Wang, Huan and Chen, Yiliu and Wu, Ziqi and Wang, Lichuan and Zou, Bin and Yang, Yafeng and Cui, Zongyang and Cai, Yu and Yu, Tianhang and Lv, Chengfei and Wu, Zhihua},
  title = {MNN: A Universal and Efficient Inference Engine},
  booktitle = {MLSys},
  year = {2020}
}
```
