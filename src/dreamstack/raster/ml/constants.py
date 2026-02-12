"""ML constants."""

import numpy as np

# ImageNet statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Caffe-style BGR mean
CAFFE_MEAN = np.array([103.939, 116.779, 123.68])
