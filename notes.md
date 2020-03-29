# Notes on [INFO8010 - Deep Learning](https://github.com/glouppe/info8010-deep-learning)

This document contains notes and additional readings for self-study.

### [Lecture 3: Computer Vision](https://glouppe.github.io/info8010-deep-learning/?p=lecture4.md)

- Misc.
  - On cross-entropy
    - [(Wiki page) Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy)
    - [A Gentle Introduction to Cross-Entropy for Machine Learning](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)
- Classification
  - Image augmentation
  - Use pre-trained models for fine tuning and transfer learning
  - Large networks trained for classification are heavily re-used for object detection and semantic segmentation tasks.
- Object Detection
  - YOLO for object detection
    - [EPFL EE-559, 8-3: Object Detection](https://fleuret.org/ee559/materials/ee559-slides-8-3-object-detection.pdf)
  - R-CNN
    - [Dive into Deep Learning - 13.8. Region-based CNNs (R-CNNs)](https://d2l.ai/chapter_computer-vision/rcnn.html)
  - Takeaways
    - One-stage detectors (YOLO, SSD, RetinaNet, etc) are *fast* for inference *not as accurate*.
    - Two-stage detectors (Fast R-CNN, Faster R-CNN, R-FCN, Light head R-CNN, etc) are usually *slower* but are *more accurate*.
    - Both depend on engineering decisions.
- Segmentation
    - Task: partitioning an image into regions of different semantic categories at *pixel level*. 
    - Fully convolutional network(FCN) and transposed convolution
      - [CS231n, Lecture 11, 2018.](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture11.pdf)
    - Mask R-CNN
      - Object detection combined with mask prediction enables instance segmentation.
      - [Dive into Deep Learning - 13.8.4 Mask R-CNN](https://d2l.ai/chapter_computer-vision/rcnn.html)

## Resrouces
- [EPFL EE-559 – Deep Learning](https://fleuret.org/ee559/) - EE-559 "Deep Learning", taught by François Fleuret in the School of Engineering of the École Polytechnique Fédérale de Lausanne, Switzerland.
- [Dive into Deep Learning](https://d2l.ai/): An interactive deep learning book with code, math, and discussions, based on the NumPy interface.