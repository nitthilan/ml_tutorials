import os

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# import get_data as gd
import tensorflow as tf

import confidence as cf
from PIL import Image


save_dir = "../../data/conv/saved_model_v2/"

resize_factor = 1
weight_path = os.path.join(save_dir, \
    "keras_cifar10_weight_"+str(resize_factor)+".h5")
json_path = os.path.join(save_dir, \
"keras_cifar10_model_"+str(resize_factor)+".json")

predict_path = os.path.join(save_dir, \
"keras_cifar10_predcit_"+str(resize_factor)+".npz")

# with tf.device("/cpu:0"):
# load the weights and model from .h5
model = load_model(weight_path)

print(weight_path, json_path, predict_path)

im = np.array(Image.open("../../data/conv/resize_16/test/0.bmp"))

print(im.shape)

im = np.expand_dims(im/255.0, axis=0)

print(model.predict(im))

print(im.shape)


with open(json_path, "w") as text_file:
	text_file.write(model.to_json())

