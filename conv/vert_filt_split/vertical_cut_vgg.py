'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
# from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import tensorflow as tf
from keras.utils import multi_gpu_model
from numpy import linalg as LA


from keras import backend as K
# import get_wide_res_networks as gwrn



import pickle
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import data_load.get_keras_data as gkd
# import get_vgg16_cifar10 as gvc
# import gen_conv_net as gcn
import conv.networks.get_all_imagenet as gai
# import SqueezeNet as sqn
# import get_vgg16_cifar10 as gvc
# import gen_conv_net as gcn
# import get_data as gd
# import get_vgg16_cifar10 as gvc


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from keras.models import load_model

# python3 conv/vert_filt_split/vertical_cut_vgg.py conv svhn 3
# python3 conv/vert_filt_split/evaluate_model.py

# List of experiments done:
# Train nets independently with different values
# Train only the last layer reusing the top layers without resizing
# Train only the last layer with resizing of the input

# Averaging and removing offset
# remove the use_bias and check whether 
# vgg16 cifar10 scale 1.33 num fikter 75% reaches 64%
# vgg16 cifar10 scale 1.5 num fikter 50% reaches 28%

# SqueezeNet
  # 12, 3, vert_filt_squeezenet_mnist_12_3_3_old.h5, 61%
  # 12, 3, vert_filt_squeezenet_mnist_2_12_1.h5, 74%

  # 4, 1, vert_filt_squeezenet_mnist_3_old.h5, 30%
  # 4, 1, vert_filt_squeezenet_mnist_2_4_1 60%

  # 4, 1, dense.h5, 42% - Not worth it since dense takes more coefficients
  # 4, 1, 62%
# MobileNet
# https://github.com/keras-team/keras/pull/9965 - transfer learning problem with BatchNormalisation

# train using Max instead of Avg pooling??

# python3 vertical_cut_vgg.py MobileNet mnist 2.5 - 30% not useful
# python3 vertical_cut_vgg.py MobileNet cifar10 2.5 - 38%
# python3 vertical_cut_vgg.py MobileNet cifar10 3.5 - 72.5%
# python3 vertical_cut_vgg.py MobileNet cifar10 3 - 55.73%
# python3 vertical_cut_vgg.py MobileNet cifar10 2 - 25.44%
# python3 vertical_cut_vgg.py SqueezeNet cifar10 3 - 12%
# python3 vertical_cut_vgg.py SqueezeNet cifar10 2 - 49%
# python3 vertical_cut_vgg.py SqueezeNet cifar10 3 - 17
# python3 vertical_cut_vgg.py MobileNet mnist 2 - 58.42%
# python3 vertical_cut_vgg.py MobileNet mnist 2.5 - 82%
# python3 vertical_cut_vgg.py MobileNet mnist 3 - 95.6%
# python3 vertical_cut_vgg.py MobileNet mnist 3.5 - 97.2%

# vgg 3 88.44,  2.25 64, 2.0 0.3875, 2.5 71.94, 3.5 0.9110, 
# MobileNet 3 80.80, 2 41.61, 2.25 0.5621, 2.5 69.08, 3.5  0.8838
# conv 4 0.9535, 2 (15), 2.5 (18) 
# MobileNet_cifar10_un_4.0 MobileNet cifar10 4 0.8580, 2 0.2540, 2.25 0.3202, 2.5 42.63%, 3 0.6254, 3.5 0.7608
# MobileNet_mnist_un_4.0 mnist 4 0.98, 2 0.7063, 3 0.9549, 2.5 .8633, 3.5 0.9716 
# vgg cifar10 4 0.8273, 2 0.4500, 3 0.7488, 2.25 0.5549, 3.5 0.8130, 2.5 0.6370
# vgg mnist 4 0.9813, 2 0.7088, 3 0.9630, 2.5 0.9352, 3.5 0.9866
# conv cifar10 4 0.8864, 2 0.3876, 3 0.7182, 2.25 0.5156, 2.5 0.5948, 3.5 0.8226
# conv mnist 4 0.9904, 2 0.9122, 3 0.9775, 2.5 0.9642, 3.5 0.9881
batch_size = 32
# num_classes = 100 # 10 for cifar10 # 100 for cifar100
epochs = 200
data_augmentation = True
save_dir = os.path.join(os.getcwd(), './conv/saved_models')
# Path for vertical filter models vert_filt_saved_models
# Path for fully trained models vert_filt_saved_keras_models

model_name = sys.argv[1] # "vgg" # "conv" # "SqueezeNet" # "MobileNet" #
dataset = sys.argv[2] # "mnist" #"cifar10" #
num_filter = float(sys.argv[3]) #2

scale = 1.0

base_weight_path = "./conv/vert_filt_saved_keras_models/"
if(model_name == "conv" and dataset == "cifar10"):
  # load_weight_path = "conv_cifar10_false_4.h5"
  load_weight_path = "conv_cifar10_un_4.0.h5"
elif(model_name == "conv" and dataset == "mnist"):
  # load_weight_path = "conv_mnist_false_4.h5"
  load_weight_path = "conv_mnist_un_4.0.h5"
elif(model_name == "conv" and dataset == "cifar100"):
  load_weight_path = "conv_cifar100_false_4.h5"
elif(model_name == "vgg" and dataset == "cifar10"):
  # load_weight_path = "vgg_cifar10_false_4.h5"
  load_weight_path = "vgg_cifar10_un_4.0.h5"
elif(model_name == "vgg" and dataset == "mnist"):
  # load_weight_path = "vgg_cifar10_false_4.h5"
  load_weight_path = "vgg_mnist_un_4.0.h5"
elif(model_name == "SqueezeNet" and dataset == "cifar10"):
  load_weight_path = "SqueezeNet_cifar10_false_20112018_1.h5"
elif(model_name == "SqueezeNet" and dataset == "mnist"):
  load_weight_path = "SqueezeNet_mnist_false_1.h5"
elif(model_name == "MobileNet" and dataset == "cifar10"):
  # load_weight_path = "MobileNet_cifar10_false_4.h5"
  load_weight_path = "MobileNet_cifar10_un_4.0.h5"
elif(model_name == "MobileNet" and dataset == "mnist"):
  # load_weight_path = "MobileNet_mnist_false_20112018_4.h5"
  load_weight_path = "MobileNet_mnist_un_4.0.h5"
elif(model_name == "MobileNet" and dataset == "svhn"):
  # load_weight_path = "MobileNet_svhn_44.0.h5"
  load_weight_path = "MobileNet_svhn_un_4.0.h5"
elif(model_name == "SqueezeNet" and dataset == "svhn"):
  # load_weight_path = "SqueezeNet_svhn_44.0.h5"
  # load_weight_path = "SqueezeNet_svhn_wbn_4.0.h5"
  load_weight_path = "SqueezeNet_svhn_sqn_4.0.h5"
elif(model_name == "ResNet50" and dataset == "svhn"):
  # load_weight_path = "ResNet50_svhn_sqn_4.0.h5"
  load_weight_path = "ResNet50_svhn_un_1.0.h5"
elif(model_name == "vgg" and dataset == "svhn"):
  # load_weight_path = "vgg_svhn_44.0.h5"
  load_weight_path = "vgg_svhn_un_4.0.h5"
  # load_weight_path = "vgg_svhn_wbn_0.5.h5"
elif(model_name == "conv" and dataset == "svhn"):
  load_weight_path = "conv_svhn_un_4.0.h5"

# elif(model_name == "vgg" and dataset == "mnist"):
#   load_weight_path = "../../data/ml_tutorial/conv/saved_model_vgg_v3/keras_cifar10_weight_0.h5"

load_weight_path = base_weight_path + load_weight_path

save_weight_path = os.path.join(save_dir, \
  "vert_filt_wbn_"+model_name+"_"+dataset+"_"+str(num_filter)+".h5")

print("Configuration")
print("=============")
print(model_name, dataset, scale, num_filter)
print(load_weight_path)
print(save_weight_path)

# Save model and weights
if not os.path.isdir(save_dir):
  os.makedirs(save_dir)


x_train, y_train, x_test, y_test = gkd.get_data(dataset)

if(model_name == "conv"):
  num_layer_to_ignore = 5
  num_weights_to_ignore = 4
elif(model_name == "vgg"):
  # x_train, x_test = gkd.scale_image(x_train, x_test)
  # num_layer_to_ignore = 1
  # num_weights_to_ignore = 7
  num_layer_to_ignore = 1
  num_weights_to_ignore = 4

elif(model_name == "SqueezeNet"):
  x_train_48 = np.zeros((x_train.shape[0], 48, 48, x_train.shape[3]))
  x_test_48 = np.zeros((x_test.shape[0], 48, 48, x_test.shape[3]))
  x_train_48[:, 8:40, 8:40, :] = x_train
  x_test_48[:, 8:40, 8:40, :] = x_test
  x_train = x_train_48
  x_test = x_test_48
  num_layer_to_ignore = 4#12#4
  num_weights_to_ignore = 1#1#3
elif(model_name == "MobileNet"):
  num_layer_to_ignore = 4#3#12#9
  num_weights_to_ignore = 2#1#11#5
elif(model_name == "ResNet50"):
  num_layer_to_ignore = 1#3#12#9
  num_weights_to_ignore = 1#1#11#5

# if(model_name == "MobileNet"):
#   x_train /= 256
#   x_test /= 256
# el
if(model_name == "conv" and dataset == "cifar100"):
  x_train /= 256
  x_test /= 256 
else:
  # Preprocessing of data
  x_train /= 128
  x_train -= 1
  x_test /= 128
  x_test -= 1


num_classes = int(y_train.shape[1])
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

with tf.device('/gpu:0'):
  trained_model = load_model(load_weight_path)
  weight_list = trained_model.get_weights()

evaluation = trained_model.evaluate(x_test, y_test)
print("Evaluation Test Set ", evaluation)
trained_model.summary()

# exit()

# if(model_name == "conv"):
#   model = gcn.get_conv_vert_net(x_train.shape[1:], num_classes, 2, num_filter,
#     use_bias=False)
# elif(model_name == "vgg"):
#   model = gvc.get_conv_vert_net(x_train.shape[1:], num_classes, num_filter,
#     use_bias=False)
# elif(model_name == "squeezenet"):
#   # model = gai.get_nets_wo_weights("SqueezeNet", num_classes, 
#   #     input_shape=x_train.shape[1:], num_filter=num_filter)
#   model = sqn.SqueezeNet(input_shape=x_train.shape[1:],
#       include_top=True, weights=None, num_filter=num_filter, 
#       use_bias=False, classes=num_classes)

model = gai.get_nets_wo_weights(model_name, num_classes, 
  input_shape=x_train.shape[1:], num_filter=num_filter, include_top=True)
new_weight_list = model.get_weights()
model.summary()

# for i, weight in enumerate(weight_list[:-1*num_weights_to_ignore]):
#   new_weight_list[i] = scale*weight[:,:,:new_weight_list[i].shape[2],:new_weight_list[i].shape[3]]
#   print("Weight info ", i, weight.shape, new_weight_list[i].shape)
#   for j in range(new_weight_list[i].shape[3]):
#     print(np.sum(new_weight_list[i][:,:,:,j]), np.sum(weight[:,:,:,j]))

for i, weight in enumerate(weight_list[:-1*num_weights_to_ignore]):
# for i, weight in enumerate(weight_list):
  print("Weight info ", i, weight.shape, new_weight_list[i].shape)
  # if(i <= 54): 65% for 75%
  # if(i >= 36): 64% for 75%
  # if(i <= 0):
  #   new_weight_list[i] = weight
  #   continue
  # else:
  #   scale = 1.0
  scale = 4.0/num_filter
  if(len(weight.shape)==4):
    if(model_name != "SqueezeNet"):
      new_weight_list[i] = scale*weight[:,:,:new_weight_list[i].shape[2],:new_weight_list[i].shape[3]]
      # if(len(weight.shape) == 4):
      #     for j in range(weight.shape[3]):
      #       print("Before ", LA.norm(weight[:,:,:,j]))
      #       weight[:,:,:,j] = weight[:,:,:,j]/LA.norm(weight[:,:,:,j])
      #       print("After ", LA.norm(weight[:,:,:,j]))

      # if(i==8):
      #   new_weight_list[i] *= 1.5
      # for j in range(new_weight_list[i].shape[3]):
      #   new_weight_list[i] *= (np.sum(weight[:,:,:,j]) / np.sum(new_weight_list[i][:,:,:,j]))
      #   # print(np.sum(new_weight_list[i][:,:,:,j]), np.sum(weight[:,:,:,j]))
    else:
      if(i in [4,7,10,13,16,19,22,25,28]):
        new_shape = new_weight_list[i].shape
        old_shape = weight.shape
        off_2 = int(new_shape[2]/2)
        # off_3 = int(new_shape[3]/2)
        off_old_2 = int(old_shape[2]/2)
        # off_old_3 = int(old_shape[3]/2)
        off_new_old_2 = int((old_shape[2]+new_shape[2])/2)
        # off_new_old_3 = int((old_shape[3]+new_shape[3])/2)
        print("Offsets 2 ", off_2, off_old_2, off_new_old_2, scale)
        # print("Offsets 3 ", off_3, off_old_3, off_new_old_3)
        new_weight_list[i][:,:,:off_2,:] = scale*weight[:,:,:off_2,:new_shape[3]]
        new_weight_list[i][:,:,off_2:,:] = \
          scale*weight[:,:,off_old_2:off_new_old_2,:new_shape[3]]
        if(len(weight.shape) == 4):
          for j in range(weight.shape[3]):
            # print("Before ", LA.norm(weight[:,:,:,j]))
            weight[:,:,:,j] = weight[:,:,:,j]/LA.norm(weight[:,:,:,j])
            # print("After ", LA.norm(weight[:,:,:,j]))
      else:
        # new_weight_list[i] = scale*weight[:,:,-new_weight_list[i].shape[2]:,-new_weight_list[i].shape[3]:]
        new_weight_list[i] = scale*weight[:,:,:new_weight_list[i].shape[2],:new_weight_list[i].shape[3]]
        # scale = np.sum(np.absolute(weight)) / np.sum(np.absolute(new_weight_list[i]))
        # new_weight_list[i] *= scale


  if(len(weight.shape)==1):
    # new_weight_list[i] = scale*weight[-new_weight_list[i].shape[0]:]
    new_weight_list[i] = scale*weight[:new_weight_list[i].shape[0]]
  if(len(weight.shape)==2):
    # new_weight_list[i] = scale*weight[-new_weight_list[i].shape[0]:, -new_weight_list[i].shape[1]:]
    new_weight_list[i] = scale*weight[:new_weight_list[i].shape[0], :new_weight_list[i].shape[1]]
  # print("The before sum ", np.sum(new_weight_list[i]), np.sum(weight),
  #     np.sum(np.absolute(new_weight_list[i])), np.sum(np.absolute(weight)))

  # scale = np.sum(np.absolute(weight)) / np.sum(np.absolute(new_weight_list[i]))
  # new_weight_list[i] *= scale
  # scale = np.sum(weight) / np.sum(new_weight_list[i])
  # new_weight_list[i] *= scale
  # print("The after sum ", np.sum(new_weight_list[i]), np.sum(weight),
  #     np.sum(np.absolute(new_weight_list[i])), np.sum(np.absolute(weight)))

model.set_weights(new_weight_list)
for layer in model.layers[:-1*num_layer_to_ignore]: # Reduce this to -1 KJN Change
  layer.trainable = False

# parallel_model = multi_gpu_model(model, gpus=4)
parallel_model = model
modelCheckpoint = ModelCheckpoint(save_weight_path, 
  monitor='val_acc', verbose=0, save_best_only=True, 
  save_weights_only=False, mode='auto', period=1)

callbacks = [
            modelCheckpoint
            #   earlyStopping, 
            #   reduceonplateau,
            #   csv_logger
            ]

# initiate RMSprop optimizer
optimizers = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
# optimizers = keras.optimizers.Adam(lr=0.001, decay=1e-6)

parallel_model.compile(loss='categorical_crossentropy', \
  optimizer=optimizers, metrics=['accuracy'])

parallel_model.summary()

if not data_augmentation:
  print('Not using data augmentation.')
  parallel_model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True,
            callbacks = callbacks)
else:
  print('Using real-time data augmentation.')
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

  # Compute quantities required for feature-wise normalization
  # (std, mean, and principal components if ZCA whitening is applied).
  datagen.fit(x_train)

  parallel_model.fit_generator(datagen.flow(x_train, y_train,
                                   batch_size=batch_size),
                      steps_per_epoch=x_train.shape[0] // batch_size,
                      epochs=epochs,
                      validation_data=(x_test, y_test),
                      callbacks = callbacks)
