import numpy as np 
from bayes_opt import BayesianOptimization
import os

import confidence as cf


# net1    net2    net3
# Avg Pow 4.723083333     4.9965  5.288769231
# time of exec    0.7     1       1.2
# EDP     2.314310833     4.9965  7.615827692
# accuracy        0.2713  0.7926  0.8186
# norm. EDP       1       2.158958048     3.290754026
# norm. acc       0.331419497     0.968238456     1


# https://github.com/fmfn/BayesianOptimization
# https://en.wikipedia.org/wiki/Multi-objective_optimization
# save_dir = "../../data/conv/saved_model_v3/"
save_dir = "../../data/conv/saved_model_v4/"


# save_dir = "../../data/conv/saved_model_vgg_v2/"
# save_dir = "./saved_models/"

predict_gen_list = []
y_test_list = []
# load all the prediction and true value
for resize_factor in [0,1,2]:
	predict_path = os.path.join(save_dir, \
    	"keras_cifar10_predcit_"+str(resize_factor)+".npz")
	loaded_data = np.load(predict_path)
	predict_gen = loaded_data['arr_0']
	y_test = loaded_data['arr_1']
	predict_gen_list.append(predict_gen)
	y_test_list.append(y_test)


def get_accuracy(ctype):
	accuracy_values = {}
	min_acc = 100
	max_acc = 0
	for idx in [0,1,2]:
		(accuracy, total_pred, conf_sel_list) = \
			cf.get_prob_based_confidence(\
				predict_gen_list[idx], \
				y_test_list[idx], \
	  			0.0, ctype)
		# print(accuracy, total_pred)
		accuracy_values[idx] = accuracy*1.0/total_pred
		# Min and max accuaracy
		if(min_acc > accuracy_values[idx]):
			min_acc = accuracy_values[idx]
		if(max_acc < accuracy_values[idx]):
			max_acc = accuracy_values[idx]
	return accuracy_values, min_acc, max_acc

def find_optimal_param(c1, c2, ctype, ld):
	dict_to_conf_map = {
		0:"max_prob", 1:"top_pred_diff", 2:"entropy"
	}
	energy_values = {
		# 0:1.0, 1:0.75, 2:0.5
		# 0:(1.0+0.75+0.5=2.25), 1:(0.75+0.5=1.25), 2:0.5
		# 0:(2.25/2.25), 1:(1.25/2.25), 2:(0.5/2.25)
		# 0:1.0, 1:0.8, 2:0.7
		# 0:0.7, 1:0.8, 2:1.0 # Inverse energy values
		# Normalised EDP values
		# 1       2.158958048     3.290754026
		# The below assumes that you rerun the full network every time
		# 0:(3.3+2.16+1), 1:(2.16+1), 2:1
		# The below assume we reuse the previous network values
		# 0: 3.3, 1: 2.16, 2: 1
		# Values calculated power values of v3 and CPU times of v4
		# 0: 2.30103, 1: 1.67777, 2: 1.0
		# Values Calculated using actual values:
		# 0.120009807, 0.4436911581, 0.8579234621
		0: 7.15, 1:3.7, 2:1.0
	}
	min_energy = 1
	max_energy = (7.15+3.7+1.0) #(3.3+2.16+1)
	ctype = int(ctype)
	accuracy_values, min_acc, max_acc = get_accuracy(ctype)


	# Passing the prediction through the last network
	# Evaluate how many images have been predicted correctly
	(accuracy_2, total_pred_2, conf_sel_list_2) = \
		cf.get_prob_based_confidence(\
			predict_gen_list[2], \
			y_test_list[2], \
  			c2, ctype)
	conf_sel_list_2 = np.logical_not(conf_sel_list_2)

	predict_gen_list_1 = predict_gen_list[1][conf_sel_list_2]
	y_test_list_1 = y_test_list[1][conf_sel_list_2]

	predict_gen_list_0 = predict_gen_list[0][conf_sel_list_2]
	y_test_list_0 = y_test_list[0][conf_sel_list_2]

	
	(accuracy_1, total_pred_1, conf_sel_list_1) = \
		cf.get_prob_based_confidence(\
			predict_gen_list_1, \
			y_test_list_1, \
  			c1, ctype)

	conf_sel_list_1 = np.logical_not(conf_sel_list_1)

	predict_gen_list_0 = predict_gen_list_0[conf_sel_list_1]
	y_test_list_0 = y_test_list_0[conf_sel_list_1]

	(accuracy_0, total_pred_0, conf_sel_list_0) = \
		cf.get_prob_based_confidence(\
			predict_gen_list_0, \
			y_test_list_0, \
  			0, ctype)

	total_act = (accuracy_2+accuracy_1+accuracy_0)/10000.0
	total_acc_nor = (total_act - min_acc)*1.0/(max_acc - min_acc)

	energy_act = \
		(energy_values[0]*total_pred_0 + \
		 energy_values[1]*total_pred_1 + \
		 energy_values[2]*total_pred_2)/10000.0
	energy_spent = (energy_act - min_energy)*1.0/(max_energy - min_energy)
	energy_spent_nor = 1 - energy_spent
	print(energy_act, energy_spent_nor, total_act, total_acc_nor)
	# print(min_energy, max_energy, min_acc, max_acc)

	# print(accuracy_2, accuracy_1, accuracy_0)
	# print(total_pred_2, total_pred_1, total_pred_0)
	# print(total_accuracy, energy_spent)

	# print(conf_sel_list_2.shape, conf_sel_list_1.shape)
	# print(predict_gen_list_1.shape, y_test_list_1.shape)
	# print(c1, c2, ctype)
	# print(np.sum(conf_sel_list_2), np.sum(conf_sel_list_1))

	return ld*total_acc_nor + (1-ld)*(energy_spent_nor)

def run_for_different_ld():
	for ld in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
		print("Weightage ", (1-ld), ld)
		bo = BayesianOptimization(find_optimal_param,
		        {'c1': (0,1), 'c2': (0,1), 'ctype':(0,0), 'ld':(ld,ld)})

		bo.maximize(init_points=5, n_iter=200, kappa=2)
		# bo.maximize(init_points=1, n_iter=2, kappa=2)

		print("Final Result ", ld, bo.res['max']['max_params'], bo.res['max']['max_val'])
		c1 = bo.res['max']['max_params']['c1']
		c2 = bo.res['max']['max_params']['c2']
		find_optimal_param(c1=c1,c2=c2,ctype=0,ld=ld)

run_for_different_ld()
# find_optimal_param(0.054468, 0.151952, 0.1, 0.0)
# find_optimal_param(0.710078, 0.706680, 0.3, 0.0)
# find_optimal_param(0.778345, 0.828118, 0.5, 0.0)
# find_optimal_param(0.786781, 0.864189, 0.7, 0.0)
# find_optimal_param(0.787741, 0.948649, 0.9, 0.0)

# find_optimal_param(1, 1,  1.0, 0.0)
# find_optimal_param(0, 1,  1.0, 0.0)
# find_optimal_param(0, 0,  1.0, 0.0)


# 0.1, {'c2': 0.010056458101672883, 'c1': 0.28101793310634904, 'ld': 0.10000000000000001, 'ctype': 0.0}, 0.90000000000000002)
# 0.3, {'c2': 0.45518450776638741, 'c1': 0.055203907708637034, 'ld': 0.29999999999999999, 'ctype': 0.0}, 0.71174871750317459)
# 0.5, {'c2': 0.7558172255037553, 'c1': 0.55293875497577882, 'ld': 0.5, 'ctype': 0.0}, 0.68957366885937699)
# 0.7, {'c2': 0.92457624696084395, 'c1': 0.74269193905685094, 'ld': 0.69999999999999996, 'ctype': 0.0}, 0.78696252336716288)
# 0.9, {'c2': 0.97397470209835324, 'c1': 0.88383360708729608, 'ld': 0.90000000000000002, 'ctype': 0.0}, 0.91932698752160102)