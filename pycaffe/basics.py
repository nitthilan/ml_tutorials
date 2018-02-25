import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

# Change the compiler to gcc-6 for it to compile
# https://chunml.github.io/ChunML.github.io/project/Installing-Caffe-Ubuntu/
# https://github.com/BVLC/caffe/wiki/GeForce-GTX-1080,---CUDA-8.0,---Ubuntu-16.04,---Caffe
# https://stackoverflow.com/questions/7832892/how-to-change-the-default-gcc-compiler-in-ubuntu
# http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html

# https://github.com/BVLC/caffe/tree/master/docker

# https://github.com/BVLC/caffe/issues/1799 - setting GPU or CPU
# https://github.com/BVLC/caffe/tree/master/examples

# docker interactive 
# docker run -ti bvlc/caffe:cpu