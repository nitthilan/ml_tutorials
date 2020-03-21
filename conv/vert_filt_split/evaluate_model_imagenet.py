from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.utils import to_categorical


from keras import backend as K
# import get_all_imagenet as gai
import pickle
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_load.get_keras_data as gkd
import conv.networks.get_all_imagenet as gai

import datetime
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from keras.models import load_model

network_type = "MobileNet"# "VGG19"# 

if network_type == "VGG19":

	load_imagenet_list = [
		"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.5.h5",
		# "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.675.h5",
		"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.625.h5",
		"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.75.h5",
		"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.875.h5",
		"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG191.0.h5"
	]
else:

	load_imagenet_list = [
		"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet0.5.h5",
		"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet0.625.h5",
		"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet0.75.h5",
		"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet0.875.h5",
		"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet1.0.h5"
	]

fraction_list = [0.5, 0.625, 0.75, 0.875, 1.0]


def get_dataset():
	train_tf_path = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/val_"+network_type.lower()+".npz"
	tf_output = np.load(train_tf_path)
	print(tf_output["arr_0"].shape, tf_output["arr_1"].shape)

	return tf_output["arr_0"], tf_output["arr_1"]


def evaluate_models(load_weight_path, x_test, y_test):
	with tf.device('/gpu:0'):
		trained_model = load_model(load_weight_path)
		weight_list = trained_model.get_weights()

	# print(load_weight_path)

	# print("Start ", datetime.datetime.now())
	evaluation = trained_model.evaluate(x_test, y_test, batch_size=128, verbose=0)
	print(evaluation)
	# print("End ",datetime.datetime.now())
	# print(model_name, dataset, num_filter, evaluation)
	# time.sleep(10)

	# trained_model.summary()
	return evaluation

def predict_models(load_weight_path, x_test, y_test):
	with tf.device('/gpu:0'):
		trained_model = load_model(load_weight_path)
		weight_list = trained_model.get_weights()

	print(load_weight_path)

	print("Start ", datetime.datetime.now())
	prediction = trained_model.predict(x_test, batch_size=128, verbose=1)
	print("End ",datetime.datetime.now())
	predict_path = load_weight_path[:-3]+"_predict.npz"
	np.savez(predict_path, prediction, y_test)
	print(predict_path)

	# print("Evaluation Test Set ", evaluation)
	# trained_model.summary()
	return

num_classes = 1000
x_test_true, y_test = get_dataset()
y_test = to_categorical(y_test, num_classes=num_classes)

if network_type == "VGG19":
	input_width = 512
else:
	input_width = 1024

for i in range(5):
	print(i)
	fraction = fraction_list[i]
	x_test = x_test_true[:,:,:,:int(fraction*input_width)]
	# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

	load_weight_path = load_imagenet_list[i]
	evaluate_models(load_weight_path, x_test, y_test)
	predict_models(load_weight_path, x_test, y_test)

# 0
# [3.6375263648986818, 0.33798, 0.607480000038147]
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.5.h5
# Start  2019-06-28 10:35:41.966925
# 50000/50000 [==============================] - 2s 44us/step
# End  2019-06-28 10:35:44.185374
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.5_predict.npz
# 1
# [5.932448180847168, 0.3775, 0.6235799999809265]
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.625.h5
# Start  2019-06-28 10:35:53.791982
# 50000/50000 [==============================] - 2s 45us/step
# End  2019-06-28 10:35:56.062242
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.625_predict.npz
# 2
# 2019-06-28 10:35:58.714521: W tensorflow/core/framework/allocator.cc:113] Allocation of 231211008 exceeds 10% of system memory.
# 2019-06-28 10:35:58.803542: W tensorflow/core/framework/allocator.cc:113] Allocation of 231211008 exceeds 10% of system memory.
# 2019-06-28 10:36:01.209022: W tensorflow/core/framework/allocator.cc:113] Allocation of 231211008 exceeds 10% of system memory.
# 2019-06-28 10:36:01.313053: W tensorflow/core/framework/allocator.cc:113] Allocation of 231211008 exceeds 10% of system memory.
# 2019-06-28 10:36:01.403114: W tensorflow/core/framework/allocator.cc:113] Allocation of 231211008 exceeds 10% of system memory.
# [3.167678414154053, 0.4233799999904633, 0.656000000038147]
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.75.h5
# Start  2019-06-28 10:36:08.928649
# 50000/50000 [==============================] - 3s 55us/step
# End  2019-06-28 10:36:11.695624
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.75_predict.npz
# 3
# [3.1414494078826904, 0.4245799999809265, 0.7001]
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.875.h5
# Start  2019-06-28 10:36:31.400069
# 50000/50000 [==============================] - 4s 81us/step
# End  2019-06-28 10:36:35.456023
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.875_predict.npz
# 4
# [1.4801151219558715, 0.6473, 0.85882]
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG191.0.h5
# Start  2019-06-28 10:36:50.582317
# 50000/50000 [==============================] - 5s 96us/step
# End  2019-06-28 10:36:55.381663
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG191.0_predict.npz


# Configuration:
# ==============
# ['VGG191.0', 'VGG190.875', 'VGG190.75', 'VGG190.625', 'VGG190.5']
# [7120.997, 4237.65, 2288.178, 1197.499, 793.75]
# {'c3': (0, 1), 'c2': (0, 1), 'c1': (0, 1), 'ctype': (0, 0), 'c4': (0, 1)}
# 1
# ===============
# (10000, 10)
# ({0: 0, 1: 0, 2: 0, 3: 0, 4: 16899}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 50000})
# (10000, 10)
# ({0: 72, 1: 31, 2: 131, 3: 2319, 4: 15508}, {0: 151, 1: 256, 2: 613, 3: 9529, 4: 39451})
# (10000, 10)
# ({0: 383, 1: 120, 2: 371, 3: 3869, 4: 14281}, {0: 855, 1: 717, 2: 1298, 3: 14320, 4: 32810})
# (10000, 10)
# ({0: 918, 1: 227, 2: 626, 3: 4988, 4: 13237}, {0: 2082, 1: 1148, 2: 1723, 3: 17236, 4: 27811})
# (10000, 10)
# ({0: 1651, 1: 333, 2: 910, 3: 5900, 4: 12135}, {0: 3747, 1: 1491, 2: 2111, 3: 18937, 4: 23714})
# (10000, 10)
# ({0: 2642, 1: 441, 2: 1149, 3: 6589, 4: 11127}, {0: 5870, 1: 1743, 2: 2333, 3: 19832, 4: 20222})
# (10000, 10)
# ({0: 3915, 1: 504, 2: 1334, 3: 7243, 4: 10060}, {0: 8433, 1: 1809, 2: 2449, 3: 20249, 4: 17060})
# (10000, 10)
# ({0: 5506, 1: 583, 2: 1474, 3: 7686, 4: 8924}, {0: 11534, 1: 1805, 2: 2442, 3: 20097, 4: 14122})
# (10000, 10)
# ({0: 7687, 1: 613, 2: 1518, 3: 8076, 4: 7618}, {0: 15470, 1: 1622, 2: 2292, 3: 19475, 4: 11141})
# (10000, 10)
# ({0: 10947, 1: 614, 2: 1441, 3: 8455, 4: 5869}, {0: 21136, 1: 1325, 2: 1920, 3: 17876, 4: 7743})
# [0.33798, 0.36122, 0.38048, 0.39992, 0.41858, 0.43896, 0.46112, 0.48346, 0.51024, 0.54652]
# [0.11146613318331688, 0.1300041589176347, 0.15528169920307505, 0.18634565542437387, 0.22280891771194397, 0.26492023036100143, 0.3120646075935715, 0.3669310441417122, 0.43377144419805264, 0.5282127740708218]



# 0
# [2.4244686387634276, 0.47398, 0.7277799999809265]
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet0.5.h5
# Start  2019-08-14 02:32:25.975643
# 50000/50000 [==============================] - 4s 80us/step
# End  2019-08-14 02:32:29.989180
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet0.5_predict.npz
# 1
# [1.9203464959716796, 0.5604400000190735, 0.799659999961853]
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet0.625.h5
# Start  2019-08-14 02:32:38.000523
# 50000/50000 [==============================] - 5s 95us/step
# End  2019-08-14 02:32:42.765431
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet0.625_predict.npz
# 2
# [1.615730640296936, 0.6188200000190734, 0.841100000038147]
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet0.75.h5
# Start  2019-08-14 02:32:51.202955
# 50000/50000 [==============================] - 6s 118us/step
# End  2019-08-14 02:32:57.103546
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet0.75_predict.npz
# 3
# 2019-08-14 02:33:01.305343: W tensorflow/core/framework/allocator.cc:113] Allocation of 22478848 exceeds 10% of system memory.
# 2019-08-14 02:33:01.640261: W tensorflow/core/framework/allocator.cc:113] Allocation of 22478848 exceeds 10% of system memory.
# 2019-08-14 02:33:01.669816: W tensorflow/core/framework/allocator.cc:113] Allocation of 22478848 exceeds 10% of system memory.
# 2019-08-14 02:33:01.692911: W tensorflow/core/framework/allocator.cc:113] Allocation of 22478848 exceeds 10% of system memory.
# 2019-08-14 02:33:01.715527: W tensorflow/core/framework/allocator.cc:113] Allocation of 22478848 exceeds 10% of system memory.
# [1.4626359731674194, 0.656100000038147, 0.865079999961853]
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet0.875.h5
# Start  2019-08-14 02:33:06.509096
# 50000/50000 [==============================] - 4s 88us/step
# End  2019-08-14 02:33:10.947440
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet0.875_predict.npz
# 4
# [1.3413404350662232, 0.6835999999809265, 0.882500000038147]
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet1.0.h5
# Start  2019-08-14 02:33:20.388626
# 50000/50000 [==============================] - 7s 134us/step
# End  2019-08-14 02:33:27.093036
# /mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet1.0_predict.npz
