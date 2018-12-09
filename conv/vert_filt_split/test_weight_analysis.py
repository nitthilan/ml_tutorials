
import os

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# import get_data as gd
import tensorflow as tf

import confidence as cf

import sys, os
from keras import backend as K
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_load.get_keras_data as gkd
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)


# https://github.com/tensorflow/models/blob/master/research/inception/inception/data/download_imagenet.sh
# imagenet validation set download
# cifar10 pretrained model keras mobilenet
# imagenet_utils keras test images

#http://web.cs.ucdavis.edu/~yjlee/teaching/ecs289g-winter2018/Pytorch_Tutorial.pdf


# save_dir = "../../data/ml_tutorial/conv/saved_model_v1/"
# save_dir = "../../data/ml_tutorial/conv/saved_model_v2/"
save_dir = "../../data/ml_tutorial/conv/saved_model_v3/"
# save_dir = "../../data/conv/saved_model_v4/"

# The model expects input to be scaled:
# It is trained using transfer learning
# save_dir = "../../data/conv/saved_model_vgg_v2"
# save_dir = "../../data/ml_tutorial/conv/saved_model_vgg_v3"
# save_dir = "../../data/conv/saved_model_wrn_v0/"
# save_dir = "../../data/conv/cifar10_conv_non_lin_v0/"

# save_dir = "../../data/conv/cifar100_vgg_v2/"
# save_dir = "../../data/conv/mnist_conv_v1/"


# save_dir = "./saved_models_1/"
batch_size = 16
num_classes = 10


# Time for V4:
# 49us/step, 63us/step, 73us/step
# 13020.6899927, 21845.7980992, 29961.0913622
# 1, 1.67777, 2.30103
# V4 in CPU:
# 3ms/step, 7ms/step, 7ms/step
 


x_train, y_train, x_test, y_test = \
  gkd.get_data("cifar10")


# x_train, y_train, x_test, y_test = \
#   gd.get_cifar_data(0, num_classes)

# x_train /= 255
# x_test /= 255
x_train, x_test = gkd.scale_image(x_train, x_test)

# x_test = x_train
# y_test = y_train

resize_factor = 0
weight_path = os.path.join(save_dir, \
  "keras_cifar10_weight_"+str(resize_factor)+".h5")
json_path = os.path.join(save_dir, \
  "keras_cifar10_model_"+str(resize_factor)+".json")

predict_path = os.path.join(save_dir, \
  "keras_cifar10_predcit_"+str(resize_factor)+".npz")

with tf.device("/gpu:0"):
# load the weights and model from .h5
  model = load_model(weight_path)
print(weight_path)
model.summary()

# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

weight_list = model.get_weights()

for i, weight in enumerate(weight_list):
  print(i, weight.shape, weight.dtype, np.min(weight), np.max(weight))


# for layer in model.layers:
  # config_list = layer.get_config()
  # print("Layer ", len())
evaluation = model.evaluate(x_test, y_test)
print("Evaluation Test Set Base ", evaluation)
weight_list_copy = copy.deepcopy(weight_list)

for i in [8]:#[10, 8, 6, 4, 2, 0]:
  eval_list = []
  weight_shape = weight_list[i].shape
  print("Num filters ", weight_shape[3])
  for j in range(weight_shape[3]):
    temp_store = copy.deepcopy(weight_list[i][:,:,:,j])
    weight_list[i][:,:,:,j] = 0
    model.set_weights(weight_list)
    evaluation = model.evaluate(x_test, y_test)
    print("Evaluation Test Set ", i, j, evaluation)
    eval_list.append(-1*evaluation[1]) 
    weight_list[i][:,:,:,j] = temp_store # restore

  list_index = np.argsort(eval_list)
  for j in list_index:
    weight_list_copy[i][:,:,:,j] = 0
    model.set_weights(weight_list_copy)
    evaluation = model.evaluate(x_test, y_test)
    print("Progressive evaluation ", i, j, evaluation)


for i,weight in enumerate(weight_list):
  weight = weight_list[i]*(2**31)
  # N = 2**20
  # weight = np.floor(weight/N)*N
  N = 10000
  weight[np.abs(weight)<N] = 0
  weight_list[i] = weight*1.0/(2**31)
  # weight_list[i] = np.floor(weight_list[i])
  print(i, N, weight.shape, weight.dtype, np.min(weight), np.max(weight))


model.set_weights(weight_list)


with open(json_path, "w") as text_file:
  text_file.write(model.to_json())

# datagen.fit(x_train)


# Evaluate model with test data set and share sample prediction results
evaluation = model.evaluate_generator(datagen.flow(x_test, y_test,
                                      batch_size=batch_size),
                                      steps=x_test.shape[0] // batch_size)

print('Model Accuracy = %.2f' % (evaluation[1]))
print("Evaluation ", evaluation)

# predict_gen = model.predict_generator(datagen.flow(x_test, y_test,
#                                       batch_size=batch_size),
#                                       steps=x_test.shape[0] // batch_size)

predict_gen = model.predict(x_test)
evaluation = model.evaluate(x_test, y_test)
print("Evaluation Test Set ", evaluation)
np.savez(predict_path, predict_gen, y_test)

conf_threshold, accuracy_list, total_pred_list = \
  cf.get_for_all_confidence(predict_gen, y_test, "max_prob")
cf.plot_confidence(conf_threshold, accuracy_list, total_pred_list,
  "max_prob_conf_"+str(resize_factor)+".png")
conf_threshold, accuracy_list, total_pred_list = \
  cf.get_for_all_confidence(predict_gen, y_test, "top_pred_diff")
cf.plot_confidence(conf_threshold, accuracy_list, total_pred_list,
  "top_pred_diff_conf_"+str(resize_factor)+".png")
conf_threshold, accuracy_list, total_pred_list = \
  cf.get_for_all_confidence(predict_gen, y_test, "entropy")
cf.plot_confidence(conf_threshold, accuracy_list, total_pred_list,
  "entropy_conf_"+str(resize_factor)+".png")

  # for predict_index, predicted_y in enumerate(predict_gen):
  #     print(predicted_y)
  #     actual_label = np.argmax(y_test[predict_index])
  #     predicted_label = np.argmax(predicted_y)
  #     print('Actual Label = %s vs. Predicted Label = %s' % (actual_label,
  #                                                           predicted_label))
  #     if predict_index == 20:
  #         break
