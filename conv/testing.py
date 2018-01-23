
import os

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

import get_data as gd
import tensorflow as tf

import confidence as cf




# One option for model conversion is:
# - Convert from keras to pytorch and then from pytorch to caffe
#   - Since there seems to be no direct conversion possibility
#     - https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/caffe/README.md
# Tools available:
# - https://github.com/ysh329/deep-learning-model-convertor
#   - https://github.com/Microsoft/MMdnn

save_dir = "../../data/conv/saved_model_v2/"
# save_dir = "./saved_models/"
batch_size = 16



for resize_factor in [0,1,2]:#[0,1,2]:
  x_train, y_train, x_test, y_test = \
    gd.get_cifar10_data(resize_factor)

  weight_path = os.path.join(save_dir, \
    "keras_cifar10_weight_"+str(resize_factor)+".h5")

  # with tf.device("/cpu:0"):
  # load the weights and model from .h5
  model = load_model(weight_path)

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

  datagen.fit(x_train)


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
