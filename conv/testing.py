
import os

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

import get_data as gd
import tensorflow as tf

import confidence as cf


# save_dir = "../../data/conv/saved_model_v1/"
# save_dir = "../../data/conv/saved_model_v2/"
save_dir = "../../data/conv/saved_model_v3/"
# save_dir = "../../data/conv/saved_model_v4/"

# The model expects input to be scaled:
# It is trained using transfer learning
save_dir = "../../data/conv/saved_model_vgg_v2"
# save_dir = "../../data/conv/saved_model_wrn_v0/"

save_dir = "./saved_models/"
batch_size = 16
num_classes = 10


# Time for V4:
# 49us/step, 63us/step, 73us/step
# 13020.6899927, 21845.7980992, 29961.0913622
# 1, 1.67777, 2.30103
# V4 in CPU:
# 3ms/step, 7ms/step, 7ms/step
 


for resize_factor in [0]:#,1,2,3]:
  x_train, y_train, x_test, y_test = \
    gd.get_cifar_data(0, num_classes)

  # x_train, y_train, x_test, y_test = \
  #   gd.get_cifar_data(0, num_classes)
  x_train, x_test = gd.scale_image(x_train, x_test)

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

  with open(json_path, "w") as text_file:
    text_file.write(model.to_json())

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

# VGG Performance
# Model Accuracy = 0.93
# ('Evaluation ', [0.47837119290828706, 0.93469999999999998])
# 10000/10000 [==============================] - 3s 344us/step
# ('Evaluation Test Set ', [0.47737391915321348, 0.93410000000000004])
# Model Accuracy = 0.86
# ('Evaluation ', [0.70741001362800593, 0.8619])
# 10000/10000 [==============================] - 2s 182us/step
# ('Evaluation Test Set ', [0.70363716936111453, 0.86509999999999998])
# ../../data/conv/saved_model_vgg_v2/keras_cifar10_weight_2.h5
# Model Accuracy = 0.74
# ('Evaluation ', [0.98500162572860717, 0.73870000000000002])
# 10000/10000 [==============================] - 2s 152us/step
# ('Evaluation Test Set ', [0.96811740074157715, 0.74450000000000005])
# ../../data/conv/saved_model_vgg_v2/keras_cifar10_weight_3.h5
# Model Accuracy = 0.53
# ('Evaluation ', [1.5076920444488526, 0.53390000000000004])
# 10000/10000 [==============================] - 1s 118us/step
# ('Evaluation Test Set ', [1.4143426961898804, 0.56859999999999999])