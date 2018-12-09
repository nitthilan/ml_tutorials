from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.applications.mobilenetv2 import preprocess_input, decode_predictions


from keras_squeezenet import SqueezeNet
from keras_applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

import numpy as np

from os import listdir
from os.path import isfile, join
import os

import matplotlib.image as mpimg
import time

# Imagenet val dataset actual prediction
# https://gist.github.com/ksimonyan/fd8800eeb36e276cd6f9#note
# http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
# https://github.com/raghakot/keras-vis/blob/master/resources/imagenet_class_index.json
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
# 	- Decode_prediction
# https://github.com/JonathanCMitchell/mobilenet_v2_keras/blob/master/test_mobilenet.py
# 	- Test file for manipulation
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py
#	- Read the file for the implementation details of the validation

# SqueezeNet :  https://arxiv.org/pdf/1602.07360.pdf
#	- https://github.com/rcmalli/keras-squeezenet/blob/master/keras_squeezenet/squeezenet.py

imagenet_path = "/mnt/additional/aryan/imagenet_validation_data/ILSVRC2012_img_val/"
# http://www.image-net.org/challenges/LSVRC/2012/
# https://cv-tricks.com/tensorflow-tutorial/keras/
# Finding actual predictions
# http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html

image_filelist = [f for f in listdir(imagenet_path) if isfile(join(imagenet_path, f))]

print("Number of files ", len(image_filelist))


# model = ResNet50(weights='imagenet')
# model = MobileNet(weights='imagenet')
# model = MobileNetV2(weights='imagenet')

model = SqueezeNet()


def get_all_prediction(image_filelist):
	prediction_list = []
	for filename in image_filelist:

		# img = image.load_img(os.path.join(imagenet_path, filename), target_size=(224, 224))
		img = image.load_img(os.path.join(imagenet_path, filename), target_size=(227, 227)) # Squeezenet
		# img1 = mpimg.imread(os.path.join(imagenet_path, filename))
		# print(img1.shape)
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = imagenet_utils.preprocess_input(x)

		preds = model.predict(x)
		# decode the results into a list of tuples (class, description, probability)
		# (one such list for each sample in the batch)
		decode_pred = imagenet_utils.decode_predictions(preds)
		print("Prediction output ", decode_pred)
		print('Predicted:', filename, imagenet_utils.decode_predictions(preds, top=3)[0])
		print("Pred values ", np.argmax(preds))
		# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
		prediction_list.append(preds)
	return prediction_list


start_time = time.time()
get_all_prediction(image_filelist[:10])
total_time = time.time() - start_time
print("Total prediction time ", total_time)

print("File list ", image_filelist[:10])