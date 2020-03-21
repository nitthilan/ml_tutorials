

import os
from PIL import Image


train_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/ILSVRC2012_img_train/"
test_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/ILSVRC2012_img_test/"
val_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/ILSVRC2012_img_val/"


val_image_list = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/caffe_ilsvrc12.tar/val.txt"
train_image_list = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/caffe_ilsvrc12.tar/train.txt"
test_image_list = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/caffe_ilsvrc12.tar/test.txt"

num_train_images = 1281166
num_val_images = 50000
num_test_images =  100000

# train and val folders have 1000 folders
# test folder has ILSVRC2012_test_00000001.JPEG 100000 images
train_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/train/"
val_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/val/"
test_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/test/"



train_folder_list = [x[0] for x in os.walk(train_folder)]
val_folder_list = [x[0] for x in os.walk(val_folder)]

print(len(train_folder_list), len(val_folder_list))

for train in train_folder_list:
	train1_folder = os.path.join(train_folder, train)
	# print(train1_folder)
	# print((os.listdir(train1_folder)))
	print(len([name for name in os.listdir(train1_folder) if os.path.isfile(os.path.join(train1_folder, name))]))
