
import os

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# import get_data as gd
import tensorflow as tf

import confidence as cf

import sys, os
from keras import backend as K

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_load.get_keras_data as gkd
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)



# save_dir = "../../data/conv/saved_model_v1/"
# save_dir = "../../data/conv/saved_model_v2/"
save_dir = "../../../data/ml_tutorial/conv/saved_model_v3/"
# save_dir = "../../data/conv/saved_model_v4/"

# The model expects input to be scaled:
# It is trained using transfer learning
# save_dir = "../../data/conv/saved_model_vgg_v2"
save_dir = "../../../data/ml_tutorial/conv/saved_model_vgg_v3"
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
 


for resize_factor in [0,1,2]:
  x_train, y_train, x_test, y_test = \
    gkd.get_data("cifar10")


  # x_train, y_train, x_test, y_test = \
  #   gd.get_cifar_data(0, num_classes)

  # x_train /= 255
  # x_test /= 255
  x_train, x_test = gkd.scale_image(x_train, x_test)

  # x_test = x_train
  # y_test = y_train


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

  # for layer in model.layers:
    # config_list = layer.get_config()
    # print("Layer ", len())
  weight_list = model.get_weights()
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

# ../../data/conv/saved_model_vgg_v2/keras_cifar10_weight_0_5.h5
# Model Accuracy = 0.92
# ('Evaluation ', [0.65001932158470155, 0.92049999999999998])
# 10000/10000 [==============================] - 3s 293us/step
# ('Evaluation Test Set ', [0.65513066682815557, 0.92269999999999996])


# ../../data/conv/saved_model_vgg_v2/keras_cifar10_weight_0.h5
# Model Accuracy = 0.93
# ('Evaluation ', [0.47330247962474825, 0.93169999999999997])
# 10000/10000 [==============================] - 76s 8ms/step
# ('Evaluation Test Set ', [0.47737328639030457, 0.93410000000000004])

# ../../data/conv/saved_model_vgg_v2/keras_cifar10_weight_1.h5
# Model Accuracy = 0.92
# ('Evaluation ', [0.65699501037597652, 0.92110000000000003])
# 10000/10000 [==============================] - 69s 7ms/step
# ('Evaluation Test Set ', [0.65513012456893915, 0.92269999999999996])

# ../../data/conv/saved_model_vgg_v2/keras_cifar10_weight_2.h5
# Model Accuracy = 0.86
# ('Evaluation ', [0.70277366542816166, 0.8609])
# 10000/10000 [==============================] - 51s 5ms/step
# ('Evaluation Test Set ', [0.70363712196350092, 0.86509999999999998])

# ../../data/conv/saved_model_vgg_v2/keras_cifar10_weight_3.h5
# Model Accuracy = 0.74
# ('Evaluation ', [0.9883854107856751, 0.73540000000000005])
# 10000/10000 [==============================] - 33s 3ms/step
# ('Evaluation Test Set ', [0.96811727981567386, 0.74450000000000005])

# ../../data/conv/saved_model_vgg_v2/keras_cifar10_weight_4.h5
# Model Accuracy = 0.53
# ('Evaluation ', [1.5028848129272461, 0.53259999999999996])
# 10000/10000 [==============================] - 19s 2ms/step
# ('Evaluation Test Set ', [1.4143427412033081, 0.56859999999999999])