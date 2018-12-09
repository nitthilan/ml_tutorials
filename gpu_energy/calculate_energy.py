from datetime import datetime
import sys


# energy_filelist = "gpu_energy_"
# time_filelist = "gpu_energy_time_"
energy_filelist = "squeezenet_gpu_energy_"
time_filelist = "squeezenet_gpu_energy_time_"
num_files = 5

def get_all_lines(filename):
	with open(filename) as file:
		output_array = file.readlines()

	return output_array

def get_time_energy_values(filename):
	energy_values = get_all_lines(filename)
	time_list = []
	energy_list = []
	for line in energy_values[1:]:
		values = line.strip().split(" ")
		# print(values[5])
		time = datetime.strptime(values[5], '%H:%M:%S.%f')
		# print(float(values[0]), time)
		energy_list.append(float(values[0]))
		time_list.append(time)
	return energy_list, time_list

def get_time_range(filename):
	time_values = get_all_lines(filename)
	# print(time_values)
	start_time_list = []
	end_time_list = []
	idx = 0
	while(idx < len(time_values)):
		values = time_values[idx].strip().split(" ")
		if(len(values) and values[0]=="Start"):
			# print(values)
			time = datetime.strptime(values[3], '%H:%M:%S.%f')
			start_time_list.append(time)
			while(idx < len(time_values)):
				values = time_values[idx].strip().split(" ")
				if(len(values) and values[0]=="End"):
					# print(values)
					time = datetime.strptime(values[3], '%H:%M:%S.%f')
					end_time_list.append(time)
					break
				idx += 1
		idx += 1
	# print(len(start_time_list), len(end_time_list))
	return start_time_list, end_time_list

def get_individual_energy_numbers(start_time_list, end_time_list, \
		energy_list, time_list):

	idx = 0
	idle_energy = 0
	idle_entries = 0

	for start_time, end_time in zip(start_time_list, end_time_list):
		print(start_time, end_time)
		total_energy = 0
		num_entries = 0
		num_lag_entries = 30

		while(idx < len(energy_list)):
			# print(time_list[idx], time_list[idx] > start_time, time_list[idx] < end_time)

			if(time_list[idx] >= start_time and time_list[idx] <= end_time \
				and energy_list[idx] > 56):
				# print("After ", energy_list[idx])
				total_energy += energy_list[idx]
				num_entries += 1
			elif(time_list[idx] > end_time and num_lag_entries != 0):
				# print("After Tail ", energy_list[idx])
				total_energy += energy_list[idx]
				num_lag_entries -= 1
				
			elif(time_list[idx] > end_time and num_lag_entries == 0):
				avg_total = total_energy/num_entries
				avg_idle = idle_energy/idle_entries
				# print(total_energy/num_entries, num_entries, idle_energy/idle_entries, idle_entries)
				print(idx, avg_total , num_entries, total_energy, avg_idle)
				# sys.exit()
				break

			else:
				idle_energy += energy_list[idx]
				idle_entries += 1
				# print("Before ", energy_list[idx])

			idx += 1

	return




for idx in [1]:#range(num_files):
	energy_file = energy_filelist+str(idx)+".txt"
	time_file = time_filelist+str(idx)+".txt"
	print(energy_file, time_file)
	energy_list, time_list = get_time_energy_values(energy_file)
	start_time_list, end_time_list = get_time_range(time_file)
	get_individual_energy_numbers(start_time_list, end_time_list, \
		energy_list, time_list)
	
