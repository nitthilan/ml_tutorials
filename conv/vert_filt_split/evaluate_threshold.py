import numpy as np 
from bayes_opt import BayesianOptimization
import os

import confidence as cf
# import get_data as gd
from PIL import Image
import sys


def get_pred_y_test(models_path_list):
	predict_gen_list = []
	y_test_list = []
	# load all the prediction and true value
	for pred_path in models_path_list:
		predict_path = pred_path[:-3]+"_predict.npz"
		loaded_data = np.load(predict_path)
		predict_gen = loaded_data['arr_0']
		y_test = loaded_data['arr_1']
		# print("Predicted value ", y_test.shape, predict_gen.shape)
		predict_gen_list.append(predict_gen)
		y_test_list.append(y_test)
	return predict_gen_list, y_test_list


def get_accuracy(num_models, ctype, predict_gen_list, y_test_list):
	accuracy_values = {}
	min_acc = 100
	max_acc = 0
	for idx in range(num_models):
		(accuracy, total_pred, conf_sel_list) = \
			cf.get_prob_based_confidence(\
				predict_gen_list[idx], \
				y_test_list[idx], \
	  			0.0, ctype)
		accuracy_values[idx] = accuracy*1.0/total_pred
		# print(accuracy, total_pred, accuracy_values[idx])

		# Min and max accuaracy
		if(min_acc > accuracy_values[idx]):
			min_acc = accuracy_values[idx]
		if(max_acc < accuracy_values[idx]):
			max_acc = accuracy_values[idx]
	return accuracy_values, min_acc, max_acc

def dump_images(test_images, output_idx):
	# print(test_images.shape[0])
	if(test_images.shape[0]<100):
		return

	N = 4
	W = 6
	image_tile = np.zeros((N*32,W*32,3))
	for i in range(N):
		for j in range(W):
			image_tile[i*32:(i+1)*32,j*32:(j+1)*32] = test_images[i*10+j]

	image_tile = 255*image_tile
	# print(image_tile.astype('uint8'))

	im = Image.fromarray(image_tile.astype('uint8'))
	im.save("image_tile_"+str(output_idx)+".png")

	return

class Optimiser:
	def __init__(self, optimiser_value, models_path_list):
		self.optimiser_value = optimiser_value
		self.models_path_list = models_path_list
		return
	def find_optimal_param(self, c1, c2, c3, ctype, ld):
		dict_to_conf_map = {
			0:"max_prob", 1:"top_pred_diff", 2:"entropy"
		}

		# num_classes = 10
		# x_train, y_train, x_test, y_test = \
	 #    	gd.get_cifar_data(0, num_classes)

		max_value = self.optimiser_value[0]
		min_value = self.optimiser_value[-1] #(3.3+2.16+1)
		# print(max_value, min_value)
		ctype = dict_to_conf_map[int(ctype)]
		predict_gen_list, y_test_list = \
			get_pred_y_test(self.models_path_list)
		accuracy_values, min_acc, max_acc = \
			get_accuracy(len(self.models_path_list),\
				ctype, predict_gen_list, y_test_list)
		# print("Accuracy ", min_acc, max_acc)

		confidence_list = [0,c1,c2,c3]
		accuracy_dict = {}
		total_pred_dict = {}
		test_image_dict = {}
		for idx in range(len(self.optimiser_value)-1, -1, -1):
			(accuracy, total_pred, conf_sel_list) = \
				cf.get_prob_based_confidence(\
					predict_gen_list[idx], \
					y_test_list[idx], \
	  				confidence_list[idx], \
	  				ctype)
			accuracy_dict[idx] = accuracy
			total_pred_dict[idx] = total_pred
			# print("Iteration ", idx)

			# 
			# dump_images(x_test[conf_sel_list], idx)
			conf_sel_list = np.logical_not(conf_sel_list)
			# x_test = x_test[conf_sel_list]

			# print(conf_sel_list)

			for idx1 in range(idx):
				predict_gen_list[idx1] = predict_gen_list[idx1][conf_sel_list]
				y_test_list[idx1] = y_test_list[idx1][conf_sel_list]

		# Total Accuracy
		total_act = 0
		value_act = 0
		for idx in range(len(self.optimiser_value)):
			total_act += accuracy_dict[idx]
			value_act += (total_pred_dict[idx]*self.optimiser_value[idx])
		total_act /= 10000.0
		value_act /= 10000.0
		total_acc_nor = (total_act - min_acc)*1.0/(max_acc - min_acc)
		
		value_spent = (value_act - min_value)*1.0/(max_value - min_value)
		value_spent_nor = 1 - value_spent
		# print(value_act, value_spent_nor, total_act, total_acc_nor)
		print(accuracy_dict, total_pred_dict)
		# print(ld*total_acc_nor + (1-ld)*(value_spent_nor))

		print_list = [c1, value_act, total_act]
		print_list = [str(a) for a in print_list]
		print(', '.join(print_list))

		return ld*total_acc_nor + (1-ld)*(value_spent_nor)

def run_for_different_ld(model_list, optimiser_values, bayes_args, ctype):
	print("Args: ", model_list, optimiser_values, bayes_args)
	# for ld in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
	# for ld in [0.1]:
	# for ld in [0.2]:
	# for ld in [0.3]:
	# for ld in [0.4]:  
	# for ld in [0.5]:
	# for ld in [0.6]:
	# for ld in [0.7]:
	# for ld in [0.8]:
	# for ld in [0.9]:
	
	# for ld in [0.9, 0.5, 0.1]:
	for ld in [0.1, 0.5, 0.9, 0.25, 0.75, 0.15, 0.35, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.45, 0.55, 0.65, 0.85, 0.95]:
	# for ld in [0.25, 0.15]:
		print("Weightage ", (1-ld), ld)
		bayes_args["ld"] = (ld, ld)
		bayes_args["ctype"] = (ctype, ctype)
		optmiser_func = Optimiser(optimiser_values, model_list)
		bo = BayesianOptimization(optmiser_func.find_optimal_param, \
			bayes_args)

		bo.maximize(init_points=5, n_iter=50, kappa=2)
		# bo.maximize(init_points=1, n_iter=2, kappa=2)

		print("Final Result ", ld, bo.res['max']['max_params'], bo.res['max']['max_val'])
		c1 = bo.res['max']['max_params']['c1']
		c2 = bo.res['max']['max_params']['c2']
		c3 = bo.res['max']['max_params']['c3']
		optmiser_func.find_optimal_param(c1=c1,c2=c2,c3=c3,ctype=ctype,ld=ld)

def run_for_fixed_threshold(model_list, optimiser_values, ctype):
	for th1 in range(11):
		th = th1*1.0/10
		# print("Threshold ", th)
		# ctype = 0
		ld = 0.25
		optmiser_func = Optimiser(optimiser_values, model_list)
		optmiser_func.find_optimal_param(c1=th,c2=th,c3=th,ctype=ctype,ld=ld)


# Order the models based on the decreasing accuracy or size
load_conv_cifar10_list = [
	"./saved_keras_models/conv_cifar10_false_4.h5",
	"./saved_models_bkup/vert_filt_conv_cifar10_3.h5",
	"./saved_models_bkup/vert_filt_conv_cifar10_2.h5",	
]
time_conv_cifar10_list = [3.373, 2.02, 1.0]#[36854, 22168, 10926]#[74, 56, 35]
time2_conv_cifar10_list = [11.38, 4.08, 1.0]#[36854, 22168, 10926]#[74, 56, 35]

load_cifar10_vgg_list = [
	"./saved_keras_models/vgg_cifar10_false_4.h5",
	"./saved_models_bkup/vert_filt_vgg16_cifar10_3.h5",
	"./saved_models_bkup/vert_filt_vgg16_cifar10_2.h5",	
	# "./saved_models_bkup/vert_filt_vgg16_cifar10_4.h5",
]
time_cifar10_vgg_list = [2.42, 1.37, 1.0]#[85210, 48287, 35087]#[105, 89, 77]
time2_cifar10_vgg_list = [5.86, 1.88, 1.0]#[85210, 48287, 35087]#[105, 89, 77]

load_conv_mnist_list = [
	"./saved_keras_models/conv_mnist_false_4.h5",
	"./saved_models_bkup/vert_filt_conv_mnist_3.h5",	
	"./saved_models_bkup/vert_filt_conv_mnist_2.h5",
]
time_conv_mnist_list = [3.54, 2.13, 1]#[37807, 22733, 10681]#[62, 50, 33]
time2_conv_mnist_list = [12.53, 4.54, 1]#[37807, 22733, 10681]#[62, 50, 33]

load_squeeze_cifar10_list = [
	"./saved_keras_models/SqueezeNet_cifar10_false_1.h5",
	"./saved_models_bkup/vert_filt_squeezenet_cifar10_2.h5",
	"./saved_models_bkup/vert_filt_squeezenet_cifar10_3.h5",
]
time_squeeze_cifar10_list = [1.44, 1.28, 1.0]#[3962, 3524, 2751]
time2_squeeze_cifar10_list = [2.08, 1.64, 1.0]#[3962, 3524, 2751]

load_squeeze_mnist_list = [
	"./saved_keras_models/SqueezeNet_mnist_false_1.h5",
	"./saved_models_bkup/vert_filt_squeezenet_mnist_2_12_1.h5",
	"./saved_models_bkup/vert_filt_squeezenet_mnist_3.h5",
]
time_squeeze_mnist_list = [1.61,1.272, 1.0]#[4377, 3456, 2716]
time2_squeeze_mnist_list = [2.59,1.62, 1.0]#[4377, 3456, 2716]

load_mobile_cifar10_list = [
	"./saved_keras_models/MobileNet_cifar10_false_4.h5",
	"./saved_models/vert_filt_MobileNet_cifar10_3.5.h5",
	"./saved_models/vert_filt_MobileNet_cifar10_3.0.h5",
	"./saved_models/vert_filt_MobileNet_cifar10_2.5.h5",
	"./saved_models/vert_filt_MobileNet_cifar10_2.0.h5",
]
time_mobile_cifar10_list = [3.40, 2.66, 1.95, 1.45, 1.0]#[7823, 6127, 4490, 3330, 2304]
time2_mobile_cifar10_list = [11.56, 7.08, 3.80, 2.10, 1.0]#

load_mobile_mnist_list = [
	"./saved_keras_models/MobileNet_mnist_false_20112018_4.h5", 
	"./saved_models/vert_filt_MobileNet_mnist_3.5.h5",
	"./saved_models/vert_filt_MobileNet_mnist_3.0.h5",
	"./saved_models/vert_filt_MobileNet_mnist_2.5.h5",
	"./saved_models/vert_filt_MobileNet_mnist_2.0.h5",
]
time_squeeze_mnist_list = [3.17, 2.57, 1.90, 1.37, 1.0]#[7538, 6119, 4514, 3267, 2381]
time2_squeeze_mnist_list = [10.1, 6.61, 3.61, 1.88, 1.0]#


load_mobile_cifar10_list_3 = [
	"./saved_keras_models/MobileNet_cifar10_false_4.h5",
	# "./saved_models/vert_filt_MobileNet_cifar10_3.5.h5",
	"./saved_models/vert_filt_MobileNet_cifar10_3.0.h5",
	# "./saved_models/vert_filt_MobileNet_cifar10_2.5.h5",
	"./saved_models/vert_filt_MobileNet_cifar10_2.0.h5",
]
time_mobile_cifar10_list_3 = [3.40, 1.95, 1.0]#[7823, 6127, 4490, 3330, 2304]
time2_mobile_cifar10_list_3 = [11.56, 3.80, 1.0]#

load_mobile_mnist_list_3 = [
	"./saved_keras_models/MobileNet_mnist_false_20112018_4.h5", 
	# "./saved_models/vert_filt_MobileNet_mnist_3.5.h5",
	"./saved_models/vert_filt_MobileNet_mnist_3.0.h5",
	# "./saved_models/vert_filt_MobileNet_mnist_2.5.h5",
	"./saved_models/vert_filt_MobileNet_mnist_2.0.h5",
]
time_mobile_mnist_list_3 = [3.17, 1.90, 1.0]#[7538, 6119, 4514, 3267, 2381]
time2_mobile_mnist_list_3 = [10.1, 3.61, 1.0]#


load_mobile_cifar10_list_4 = [
	"./saved_keras_models/MobileNet_cifar10_false_4.h5",
	"./saved_models/vert_filt_MobileNet_cifar10_3.5.h5",
	"./saved_models/vert_filt_MobileNet_cifar10_3.0.h5",
	"./saved_models/vert_filt_MobileNet_cifar10_2.5.h5",
	# "./saved_models/vert_filt_MobileNet_cifar10_2.0.h5",
]
time_mobile_cifar10_list_4 = [3.40, 2.66, 1.95, 1.45]#[7823, 6127, 4490, 3330, 2304]
time2_mobile_cifar10_list_4 = [11.56, 7.08, 3.80, 2.10]#

load_mobile_mnist_list_4 = [
	"./saved_keras_models/MobileNet_mnist_false_20112018_4.h5", 
	"./saved_models/vert_filt_MobileNet_mnist_3.5.h5",
	"./saved_models/vert_filt_MobileNet_mnist_3.0.h5",
	"./saved_models/vert_filt_MobileNet_mnist_2.5.h5",
	# "./saved_models/vert_filt_MobileNet_mnist_2.0.h5",
]
time_mobile_mnist_list_4 = [3.17, 2.57, 1.90, 1.37]#[7538, 6119, 4514, 3267, 2381]
time2_mobile_mnist_list_4 = [10.1, 6.61, 3.61, 1.88]#



# config_list = [
# #0
# [load_conv_cifar10_list, time_conv_cifar10_list],
# #1
# [load_cifar10_vgg_list, time_cifar10_vgg_list],
# #2
# [load_conv_mnist_list, time_conv_mnist_list],
# #3
# [load_squeeze_cifar10_list, time_squeeze_cifar10_list],
# #4
# [load_squeeze_mnist_list, time_squeeze_mnist_list],
# ]

config_list = [
#0
[load_conv_cifar10_list, time2_conv_cifar10_list],
#1
[load_cifar10_vgg_list, time2_cifar10_vgg_list],
#2
[load_conv_mnist_list, time2_conv_mnist_list],
#3
[load_squeeze_cifar10_list, time2_squeeze_cifar10_list],
#4
[load_squeeze_mnist_list, time2_squeeze_mnist_list],
#5
[load_mobile_cifar10_list_3, time2_mobile_cifar10_list_3],
#6
[load_mobile_mnist_list_3, time2_mobile_mnist_list_3],
#7
[load_mobile_cifar10_list_4, time2_mobile_cifar10_list_4],
#8
[load_mobile_mnist_list_4, time2_mobile_mnist_list_4],
]


dict_to_conf_map = {
	0:"max_prob", 1:"top_pred_diff", 2:"entropy"
}


config = int(sys.argv[1])
confidence_type = int(sys.argv[2])
num_thresh = int(sys.argv[3])

if(num_thresh == 2):
	bayes_args = {'c1': (0,1), 'c2': (0,1), 'c3': (0,0), 'ctype':(0,0)}
else:
	bayes_args = {'c1': (0,1), 'c2': (0,1), 'c3': (0,1), 'ctype':(0,0)}

print("Configuration:")
print("==============")
print(config_list[config][0])
print(config_list[config][1])
print(bayes_args)
print(confidence_type)
print("===============")

# run_for_different_ld(config_list[config][0], \
# 	config_list[config][1], bayes_args, confidence_type)

# run_for_different_ld(1)
# run_for_different_ld(2)

run_for_fixed_threshold(config_list[config][0],\
	config_list[config][1], confidence_type)

# find_optimal_param(0.898267322, 0.954418224, 0.99753636, 0.95, 0.0)
# (19.448779699999999, 0.61551425088052025, 0.9204, 0.99350465970064961)
# ({0: 2507, 1: 5361, 2: 1336, 3: 0}, {0: 3144, 1: 5517, 2: 1339, 3: 0})

# find_optimal_param(0.9738, 0.9546, 0, 0.95, 0.0)
# (4.1377209423874364, 0.3203954048854929, 0.8982, 1.0009638554216866)
# ({0: 3871, 1: 4848, 2: 263}, {0: 4826, 1: 4911, 2: 263})

# find_optimal_param(0.0, 0.0, 0, 0.1, 0.0)
# find_optimal_param(0.0, 0.44, 0, 0.3, 0.0)
# find_optimal_param(0.418631, 0.6821, 0, 0.5, 0.0)
# find_optimal_param(0.812, 0.7503, 0, 0.7, 0.0)
# find_optimal_param(0.8883, 0.9421, 0, 0.9, 0.0)

# (1.0, 1.0, 0.6905, 0.0)
# ({0: 0, 1: 0, 2: 6905}, {0: 0, 1: 0, 2: 10000})
# (1.851966, 0.81547115912711621, 0.81189999999999996, 0.58506024096385512)
# ({0: 0, 1: 3489, 2: 4630}, {0: 0, 1: 4600, 2: 5400})
# (2.6827498244236478, 0.63553020357626433, 0.8679, 0.8549397590361445)
# ({0: 763, 1: 5414, 2: 2502}, {0: 1168, 1: 6174, 2: 2658})
# (3.346775461961796, 0.49170770220321403, 0.89159999999999995, 0.96915662650602374)
# ({0: 2289, 1: 4759, 2: 1868}, {0: 3086, 1: 4978, 2: 1936})
# (3.8058318167681442, 0.3922798646513086, 0.89670000000000005, 0.99373493975903626)
# ({0: 2826, 1: 5763, 2: 378}, {0: 3704, 1: 5916, 2: 380})

# find_optimal_param(1, 1,  1.0, 0.0)
# find_optimal_param(0, 1,  1.0, 0.0)
# find_optimal_param(0, 0,  1.0, 0.0)


# 0.1, {'c2': 0.010056458101672883, 'c1': 0.28101793310634904, 'ld': 0.10000000000000001, 'ctype': 0.0}, 0.90000000000000002)
# 0.3, {'c2': 0.45518450776638741, 'c1': 0.055203907708637034, 'ld': 0.29999999999999999, 'ctype': 0.0}, 0.71174871750317459)
# 0.5, {'c2': 0.7558172255037553, 'c1': 0.55293875497577882, 'ld': 0.5, 'ctype': 0.0}, 0.68957366885937699)
# 0.7, {'c2': 0.92457624696084395, 'c1': 0.74269193905685094, 'ld': 0.69999999999999996, 'ctype': 0.0}, 0.78696252336716288)
# 0.9, {'c2': 0.97397470209835324, 'c1': 0.88383360708729608, 'ld': 0.90000000000000002, 'ctype': 0.0}, 0.91932698752160102)