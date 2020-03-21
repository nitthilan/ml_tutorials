import tensorflow as tf
import sys, os
from keras import backend as K



model_list = [
	"./saved_keras_models/conv_cifar10_false_4.h5",
	"./saved_models_bkup/vert_filt_conv_cifar10_2.h5",
	"./saved_models_bkup/vert_filt_conv_cifar10_3.h5",
	"./saved_keras_models/vgg_cifar10_false_4.h5",
	"./saved_models_bkup/vert_filt_vgg16_cifar10_2.h5",
	"./saved_models_bkup/vert_filt_vgg16_cifar10_3.h5",
	"./saved_models_bkup/vert_filt_vgg16_cifar10_4.h5",
	"./saved_keras_models/conv_mnist_false_4.h5",
	"./saved_models_bkup/vert_filt_conv_mnist_2.h5",
	"./saved_models_bkup/vert_filt_conv_mnist_3.h5",
	"./saved_keras_models/SqueezeNet_cifar10_false_1.h5",
	"./saved_models_bkup/vert_filt_squeezenet_cifar10_2.h5",
	"./saved_models_bkup/vert_filt_squeezenet_cifar10_3.h5",
	"./saved_keras_models/SqueezeNet_mnist_false_1.h5",
	"./saved_models_bkup/vert_filt_squeezenet_mnist_2_12_1.h5",
	"./saved_models_bkup/vert_filt_squeezenet_mnist_3.h5",
]

model_list = [
	"./saved_keras_models/MobileNet_mnist_false_20112018_4.h5",
	"./saved_models/vert_filt_MobileNet_mnist_3.0.h5",
	"./saved_models/vert_filt_MobileNet_mnist_2.5.h5",
	"./saved_models/vert_filt_MobileNet_mnist_2.0.h5",
	"./saved_models/vert_filt_MobileNet_mnist_3.5.h5",

	"./saved_keras_models/MobileNet_cifar10_false_4.h5",
	"./saved_models/vert_filt_MobileNet_cifar10_2.0.h5",
	"./saved_models/vert_filt_MobileNet_cifar10_2.5.h5",
	"./saved_models/vert_filt_MobileNet_cifar10_3.0.h5",
	"./saved_models/vert_filt_MobileNet_cifar10_3.5.h5",

	"./saved_keras_models/SqueezeNet_cifar10_false_20112018_1.h5",
	"./saved_models/vert_filt_SqueezeNet_cifar10_2.0.h5",
	"./saved_models/vert_filt_SqueezeNet_cifar10_3.0.h5",

]

model_list = [
"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.5.h5",
"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG191.0.h5",
"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.75.h5",
"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.625.h5",
"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG190.875.h5",

]

model_list = [
"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet_for_mobile0.625.h5",
"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet_for_mobile0.875.h5",
"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet_for_mobile0.75.h5",
"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet_for_mobile0.5.h5",
"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/MobileNet_for_mobile1.0.h5",

] 
model_list = [
"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG19_for_mobile0.625.h5",
"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG19_for_mobile0.875.h5",
"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG19_for_mobile0.75.h5",
"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG19_for_mobile0.5.h5",
"/mnt/additional/nitthilan/data/ml_tutorial/imagenet/VGG19_for_mobile1.0.h5",

] 

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto(allow_soft_placement = True)
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)

for model in model_list:
	# input_keras_model = "../conv/"+model
	input_keras_model = model

	output_tflite_model = input_keras_model[:-3]+".tflite"
	converter = tf.contrib.lite.TocoConverter.from_keras_model_file(input_keras_model)
	tflite_model = converter.convert()
	open(output_tflite_model, "wb").write(tflite_model)