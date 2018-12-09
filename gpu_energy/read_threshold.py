from datetime import datetime
import sys


# energy_filelist = "gpu_energy_"
# time_filelist = "gpu_energy_time_"
time_filelist = "../conv/thres_perf/output_" 
energy_filelist = "../conv/thres_perf/edp_output_"
num_files = 5

def get_all_lines(filename):
	with open(filename) as file:
		output_array = file.readlines()

	return output_array

def get_final_result(filename_list, file_list, num_thresh_list, offset):
	for i, j in zip(file_list, num_thresh_list):
		filename = filename_list+str(i)+"_0_"+str(j)+".txt"
		print(filename)
		lambda_list = []
		accuracy_list =[]
		value_list = []
		c1_list = []
		c2_list = []
		lines = get_all_lines(filename)
		for idx, line in enumerate(lines):
			words = line.strip().split(",")
			if(words[0] == "('Weightage '"):
				lam_da_0 = float(words[1])
				lam_da_1 = float(words[2][:-1])
			# print(words)
			if(words[0] == "('Final Result '"):
				c1 = float(words[3][6:])
				c2 = float(words[4][6:])
				words = lines[idx+offset].strip().split(",")
				if(offset != 2):
					value = float(words[0][1:])
				else:
					value = float(words[1])
				accuracy = float(words[2])
				print_list = [lam_da_1, value, accuracy, c1, c2]
				print_list = [str(a) for a in print_list]
				print(', '.join(print_list))
				lambda_list.append(lam_da_1)
				accuracy_list.append(accuracy)
				c1_list.append(c1)
				c2_list.append(c2)
				value_list.append(value)
	return lambda_list, accuracy_list, value_list

# lambda_list, accuracy_list, value_list = \
# 	get_final_result(time_filelist, range(num_files), [2, 2, 2, 2], 1)

lambda_list, accuracy_list, value_list = \
	get_final_result(energy_filelist, range(num_files), [2, 2, 2, 2], 5)

lambda_list, accuracy_list, value_list = \
	get_final_result(energy_filelist, [5, 6, 7, 8], [2, 2, 3, 3], 2)
