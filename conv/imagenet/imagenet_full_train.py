from keras_preprocessing.image import ImageDataGenerator
import keras
import pandas as pd
import tensorflow as tf
from keras import backend as K
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import data_load.get_keras_data as gkd
import conv.networks.get_all_imagenet as gai

# from keras.applications.mobilenet import preprocess_input
import numpy as np




# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

train_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/train/"
test_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/test/"
val_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/val/"

# train_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet-sz/224/train/"
# test_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet-sz/224/test/"
# val_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet-sz/224/val/"


val_image_list = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/caffe_ilsvrc12.tar/val.txt"
train_image_list = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/caffe_ilsvrc12.tar/train.txt"
test_image_list = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/caffe_ilsvrc12.tar/test.txt"
num_train_images = 1281166
num_val_images =     50000
num_test_images =  100000

img_dim = 224
network_type = "MobileNet"#"VGG19"#"MobileNetV2"
num_epochs = 100000
batch_size = 128



df_train=pd.read_csv(train_image_list, delim_whitespace=True, header=None, names=["filename", "class"])
df_val=pd.read_csv(val_image_list, delim_whitespace=True, header=None, names=["filename", "class"])
df_test=pd.read_csv(test_image_list, delim_whitespace=True, header=None, names=["filename", "class"])
# df_train = df_train.apply(lambda x: x.astype(str).str.lower())
# df_val = df_val.apply(lambda x: x.astype(str).str.lower())
# df_test = df_test.apply(lambda x: x.astype(str).str.lower())

# df_train["class"] += 1
# df_val["class"] += 1


# print(df_test)
# print(df_val)
# print(df_train)
preprocess_input = gai.preprocess_image_fn(network_type)
train_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(df_train,
	directory=train_base_folder, 
	x_col='filename', y_col='class', shuffle=False, target_size=(img_dim, img_dim), has_ext=True, batch_size=batch_size)
	# , color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, 
	# shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png',
	# subset=None, interpolation='nearest', drop_duplicates=True)
valid_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(df_val, 
	directory=val_base_folder, 
	x_col='filename', y_col='class', shuffle=False, target_size=(img_dim, img_dim), has_ext=True, batch_size=batch_size)

train_model = gai.get_all_nets(network_type)


# weight_path = os.path.join(save_dir, \
# 	model_name+str(num_filter)+".h5")
# 	# model.save(weight_path)

# modelCheckpoint = ModelCheckpoint(weight_path, 
# 	monitor='val_acc', verbose=0, save_best_only=True, 
# 	save_weights_only=False, mode='auto', period=1)

callbacks = [
          # modelCheckpoint
          #   earlyStopping, 
          #   reduceonplateau,
          #   csv_logger
          ]

# initiate RMSprop optimizer
# optimizers = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
# optimizers = keras.optimizers.Adam(lr=0.01, decay=1e-6)
optimizers = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

train_model.compile(loss='categorical_crossentropy', \
	optimizer=optimizers, metrics=['accuracy'])

# train_model.fit_generator(train_generator,
#                   steps_per_epoch=num_train_images // batch_size,
#                   epochs=num_epochs,
#                   validation_data=valid_generator,
#                   validation_steps=num_val_images // batch_size,
#                   callbacks = callbacks)

result = train_model.evaluate_generator(train_generator,
                steps=20, #50000 // batch_size, 
                # callbacks=callbacks, 
                max_queue_size=10, 
                workers=2, use_multiprocessing=False, verbose=1)
print(result)
predict = train_model.predict_generator(train_generator, 
	steps=20, 
	# callbacks=None, 
	max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
# print("To category ", np.argmax(predict, axis=1))
predict = np.argmax(predict, axis=1)

num_correct = 0
for pred, pred1 in zip(predict,df_train['class'][:640]):
	if(pred == pred1):
		num_correct += 1
	print("Predcition ", pred, pred1)
print(num_correct/640.0)




# result = train_model.evaluate_generator(valid_generator,
#                 steps=10, #num_val_images // batch_size, 
#                 # callbacks=callbacks, 
#                 max_queue_size=10, 
#                 workers=1, use_multiprocessing=False, verbose=1)

# print(result)

# predict = train_model.predict_generator(valid_generator, 
# 	steps=20, 
# 	# callbacks=None, 
# 	max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
# print("To category ", np.argmax(predict, axis=1))
# predict = np.argmax(predict, axis=1)

# num_correct = 0
# for pred, pred1 in zip(predict,df_val['class'][:640]):
# 	if(pred == pred1):
# 		num_correct += 1
# 	print("Predcition ", pred, pred1)
# print(num_correct/640.0)





# ILSVRC2012_val_00049933.JPEG 65
# ILSVRC2012_val_00049934.JPEG 222
# ILSVRC2012_val_00049935.JPEG 646
# ILSVRC2012_val_00049936.JPEG 391
# ILSVRC2012_val_00049937.JPEG 100
# ILSVRC2012_val_00049938.JPEG 521
# ILSVRC2012_val_00049939.JPEG 252
# ILSVRC2012_val_00049940.JPEG 535
# ILSVRC2012_val_00049941.JPEG 787