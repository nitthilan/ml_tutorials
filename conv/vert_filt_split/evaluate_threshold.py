import threshold_optimiser as to
from bayes_opt import BayesianOptimization
import os
import sys
import numpy as np

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
	for ld in [0.95, 0.9, 0.5, 0.25, 0.75, 0.15, 0.35, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.45, 0.55, 0.65, 0.85, 0.1]:
	# for ld in [0.25, 0.15]:
		print("Weightage ", (1-ld), ld)
		bayes_args["ld"] = (ld, ld)
		bayes_args["ctype"] = (ctype, ctype)
		optmiser_func = to.Optimiser(optimiser_values, model_list)
		bo = BayesianOptimization(optmiser_func.find_optimal_param, \
			bayes_args)

		# bo.maximize(init_points=5, n_iter=50, kappa=2)
		bo.maximize(init_points=5, n_iter=50, kappa=2)
		# bo.maximize(init_points=1, n_iter=2, kappa=2)

		print("Final Result ", ld, bo.res['max']['max_params'], bo.res['max']['max_val'])
		c1 = bo.res['max']['max_params']['c1']
		c2 = bo.res['max']['max_params']['c2']
		c3 = bo.res['max']['max_params']['c3']
		c4 = bo.res['max']['max_params']['c4']
		optmiser_func.find_optimal_param(c1=c1,c2=c2,c3=c3,c4=c4,ctype=ctype,ld=ld)

def run_for_fixed_threshold(model_list, optimiser_values, ctype, num_thresh):

	# for th1 in range(90,101):
	# 	th = th1/100.0
	N = 10.0
	N1 = 0
	optmiser_func = to.Optimiser(optimiser_values, model_list, False)
	# optmiser_func.print_model_stats(ctype)
	
	accuracy_list = []
	energy_list = []
	# range_list = np.array(range(int(N)))/N
	# range_list_1 = np.array(range(10*int(N-1), 10*int(N)))/(10*N)
	# range_list = np.concatenate((range_list, range_list_1))
	range_list = np.array(range(int(N)))/N
	for th in range_list:
		# print("Threshold ", th)
		# ctype = 0
		# ld = 0.25
		if(num_thresh == 1):
			c1 = 1.1; c2=1.1; c3=1.1; c4=th;
		elif(num_thresh == 2):
			c1 = 1.1; c2=th; c3=1.1; c4=th;
		elif(num_thresh == 3):
			c1 = th; c2=1.1; c3=th; c4=th;
		else:
			c1 = th; c2=th; c3=th; c4=th;

		# optmiser_func.find_optimal_param(c1=th,c2=th,c3=1.1,c4=1.1,ctype=ctype,ld=ld)
		(accuracy, energy) = optmiser_func.find_acc_energy(c1=c1,c2=c2,c3=c3,c4=c4,ctype=ctype)
		accuracy_list.append(accuracy)
		energy_list.append(energy)
		# print(th, accuracy, energy)
	print(accuracy_list)
	print(energy_list)




dict_to_conf_map = {
	0:"max_prob", 1:"top_pred_diff", 2:"entropy"
}

config_list = to.config_list
BASE_FOLDER = to.BASE_FOLDER
config = int(sys.argv[1])
confidence_type = int(sys.argv[2])
num_thresh = int(sys.argv[3])

# if(num_thresh == 1):
# 	bayes_args = {'c1': (0,1), 'c2': (0,0), 'c3': (0,0), 'c4': (0,0), 'ctype':(0,0)}
# elif(num_thresh == 2):
# 	bayes_args = {'c1': (0,1), 'c2': (0,1), 'c3': (0,0), 'c4': (0,0), 'ctype':(0,0)}
# elif(num_thresh == 3):
# 	bayes_args = {'c1': (0,1), 'c2': (0,1), 'c3': (0,1), 'c4': (0,0), 'ctype':(0,0)}
# else:
# 	bayes_args = {'c1': (0,1), 'c2': (0,1), 'c3': (0,1), 'c4': (0,1), 'ctype':(0,0)}


skip_thresh = (1.1,1.1)
ctype = (confidence_type, confidence_type)
ctype = (0, 0)
if(num_thresh == 1):
	bayes_args = {'c1': skip_thresh, 'c2': skip_thresh, 'c3': skip_thresh, 'c4': (0,1), 'ctype':ctype}
elif(num_thresh == 2):
	bayes_args = {'c1': skip_thresh, 'c2': (0,1), 'c3': skip_thresh, 'c4': (0,1), 'ctype':ctype}
elif(num_thresh == 3):
	bayes_args = {'c1': (0,1), 'c2': skip_thresh, 'c3': (0,1), 'c4': (0,1), 'ctype':ctype}
else:
	bayes_args = {'c1': (0,1), 'c2': (0,1), 'c3': (0,1), 'c4': (0,1), 'ctype':ctype}

print("Configuration:")
print("==============")
print(config_list[config][0])
print(config_list[config][1])
print(bayes_args)
print(confidence_type)
print("===============")

for i in range(len(config_list[config][0])):
	config_list[config][0][i] = os.path.join(BASE_FOLDER, config_list[config][0][i])

# run_for_different_ld(config_list[config][0], \
# 	config_list[config][1], bayes_args, confidence_type)

# run_for_different_ld(1)
# run_for_different_ld(2)
run_for_fixed_threshold(config_list[config][0],\
	config_list[config][1], confidence_type, num_thresh)

# for config in range(9):
# 	for i in range(len(config_list[config][0])):
# 		config_list[config][0][i] = os.path.join(BASE_FOLDER, config_list[config][0][i])

# 	run_for_fixed_threshold(config_list[config][0],\
# 		config_list[config][1], confidence_type, num_thresh)
