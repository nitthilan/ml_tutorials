import os, errno
# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import stat
import subprocess
import networkx as nx
import pandas as pd
import data_load.link_distribution_16 as ld16
import data_load.utility_functions as U
import numpy as np
import data_load.constants as C

ld = ld16.LinkDistribution()    


reference_3d_mesh = {
	"canneal":{ "latency": 17.194, "energy": 2.18E-09,"edp": 3.74E-08},
	"dedup":{ "latency": 16.926, "energy": 2.15E-09, "edp": 3.63E-08},
	"fft":{ "latency": 18.227, "energy": 2.24E-09, "edp": 4.08E-08},
	"fluid":{"latency": 18.184, "energy": 2.24E-09, "edp": 4.07E-08},
	"lu":{"latency": 17.269, "energy": 2.17E-09, "edp": 3.75E-08},
	"radix":{"latency": 18.17, "energy": 2.22E-09, "edp": 4.04E-08},
	"vips":{"latency":  17.513, "energy": 2.19E-09, "edp": 3.84E-08},
	"water":{"latency":  15.1, "energy": 1.97E-09, "edp": 2.98E-08}
}

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

class Simulator(object):
	def __init__(self, sim_path, 
		sim_name,
		opt_val_idx = 2 # 0-Latency, 1-Energy, 2-EDP
		):
		self.sim_path = sim_path
		self.opt_val_idx = opt_val_idx
		self.sim_name = sim_name
		return
	# Generate the sw_connections.txt file require for the simulator
	def generate_sw_connections(self, connection_idx_list, path):
		sw_matrix = -1*np.ones((C.SW_MAT_DIM, C.SW_MAT_DIM))
		# Generate the router to core and core to router connection
		for i in range(C.NUM_CORES):
			sw_matrix[i, C.NUM_CORES+i] = 2
			sw_matrix[C.NUM_CORES+i, i] = 2
		# Set the core connection information based on the node connection list
		for connection_idx in connection_idx_list:
			(start_index, end_index) = connection_idx
			sw_matrix[C.NUM_CORES+start_index, C.NUM_CORES+end_index] = 1
			sw_matrix[C.NUM_CORES+end_index, C.NUM_CORES+start_index] = 1
		np.savetxt(path, sw_matrix, fmt='%1d', delimiter='\t', newline='\n', header='', footer='', comments='# ')
		#print("Savetxt completed", sw_matrix.shape, path)
		return

	# Not used but given a sw_connection.txt path it reads the list of connections in it
	def get_node_connection_list(self, sw_connection_path):
		sw_connection = pd.read_csv(sw_connection_path, delim_whitespace=True, header=None).as_matrix()
		node_connection_list = []
		for i in range(C.NUM_CORES):
			for j in range(i+1, C.NUM_CORES):
				if(sw_connection[C.NUM_CORES+i, C.NUM_CORES+j] == 1):
					node_connection_list.append((U.get_node_position(i), U.get_node_position(j)))
					# print((i,j), (get_node_position(i), get_node_position(j)))
				if(sw_connection[C.NUM_CORES+i, C.NUM_CORES+j] != sw_connection[C.NUM_CORES+j, C.NUM_CORES+i]):
					print("Matrix not symetric ", i, j)

		print("Num router connections ", len(node_connection_list))
		return (node_connection_list)


	def _run_conn_list(self, connection_idx_list):
		# Remove "sw_connection.txt" if it exists
		sw_connection_path = os.path.join(self.sim_path, "sw_connection.txt")
		silentremove(sw_connection_path)
		self.generate_sw_connections(connection_idx_list, sw_connection_path)
		# print(sw_connection_path, feature_vector.shape, len(connection_idx_list))

		p = subprocess.Popen(["./NoCsim"], shell=True, 
				cwd=self.sim_path, 
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE)
		# p.wait()
		out, err = p.communicate()
		if(p.returncode):
			print("Error in running simulator ", p.returncode, err)
			return [0, 0, 0]
		# sprint ("Communicate output ", out, err, p.returncode)
		
		# print("Sim Path", self.sim_path)
		# output = p.stdout.read()
		# print ("Exe output ", output)

		# exit_status = os.system(nocsim_path)
		# print(nocsim_path, exit_status)

		# Read the output.txt for result
		output_file = os.path.join(self.sim_path, "output.txt")
		with open(output_file) as f:
			content = f.readlines()

		content = [float(x.strip().split()[1]) for x in content] # [Latency, Energy, EDP]
		# print content
		return content

	def _run_mesh(self):
		conn_idx_list = U.generate_mesh_link_list()
		return self._run_conn_list(conn_idx_list)

	def _run(self, feature_vector):
		connection_idx_list = ld.generate_conn_idx_list(feature_vector)
		# print(connection_idx_list)
		return self._run_conn_list(connection_idx_list)

		

		# ds = Dummy_Simulator(os.path.join(self.sim_path, "input_benchmark.txt"))
		# return ds.run(feature_vector)

	# Run simulator and return the particular value for Baysian Optimisation
	def run(self, feature_vector):
		result = self._run(feature_vector)
		idx_to_val_map = {0:"latency", 1:"energy", 2:"edp"}
		ref_val_to_scale = reference_3d_mesh[self.sim_name][idx_to_val_map[self.opt_val_idx]]
		# Calculating the latency compensated EDP
		scaled_latency = (result[0]-6)/(reference_3d_mesh[self.sim_name]["latency"] - 6)
		latency_compensated_edp = scaled_latency*result[1]*result[0]
		scaled_result =  100*(1-(latency_compensated_edp/reference_3d_mesh[self.sim_name]["edp"]))
		print(result, scaled_latency, latency_compensated_edp, reference_3d_mesh[self.sim_name]["edp"], scaled_result)
		return scaled_result

	def run_sim_for_max(self, max_val_filename, act_sim_output_file):
		feature_vector_list = pd.read_csv(max_val_filename, delim_whitespace=True, header=None).as_matrix()
		# print(feature_vector[0,1:])
		result = []
		max_val = -999999
		for feature_vector in feature_vector_list:
			iteration = feature_vector[0]
			dummy_sim_value = feature_vector[1]
			# Run onlu when the max value changes
			if(max_val < dummy_sim_value):
				act_sim_value = self._run(feature_vector[2:])
				max_val = dummy_sim_value
			feature_result = [iteration, dummy_sim_value]
			feature_result.extend(act_sim_value)
			result.append(feature_result)
			output_result = np.array(result)
			# print(output_result.shape, output_result)
			np.savetxt(act_sim_output_file, output_result, 
				fmt='%d %2.4f %1.4e %1.4e %1.4e ', delimiter='\t', newline='\n', header='', 
				footer='', comments='# ')

		return result



def call_simulator(sim_path, sim_run_path):
	# print(sim_path, sim_run_path)
	file_list = ["input_benchmark.txt", "NoCsim", "size.txt", "sw_connection.txt"]
	temp_folder = os.path.join(sim_run_path, "temp_folder")

	# Check if folder exits and delete it
	# Create a new folder for running the simulator
	if(os.path.isdir(temp_folder)):
		shutil.rmtree(temp_folder)
	os.makedirs(temp_folder, 0777)
	if(not os.path.isdir(temp_folder)):
		print ("Error in creating temp folder. Exiting ..")
		return
	
	# Copy the simulator and its related files	
	for file in file_list:
		src_file_path = os.path.join(sim_path, file)
		dst_file_path = os.path.join(temp_folder, file)
		shutil.copyfile(src_file_path, dst_file_path)

	# Call simulator
	nocsim_path = os.path.join(temp_folder, "NoCsim")
	os.chmod(nocsim_path, 0o777)
	temp_folder_full_path = os.path.dirname(os.path.realpath(nocsim_path))
	p = subprocess.Popen(["./NoCsim"], shell=True, cwd=temp_folder_full_path, stdout=subprocess.PIPE)
	p.wait()
	print(temp_folder_full_path)
	output = p.stdout.read()
	# print ("Exe output ", output)

	# exit_status = os.system(nocsim_path)
	# print(nocsim_path, exit_status)

	# Read the output.txt for result
	output_file = os.path.join(temp_folder, "output.txt")
	with open(output_file) as f:
		content = f.readlines()

	content = [float(x.strip().split()[1]) for x in content] # [Latency, Energy, EDP]
	# print content


	return content

# Run all the simulators by copying them to a temp folder and then running it
def run_all_benchmarks(sim_path, sim_run_path):
	dir_list = []
	for directory in os.listdir(sim_path):
		if(directory != ".DS_Store" and directory != "contains.txt"):
			dir_list.append(directory)
	# print  dir_list
	for benchmark in dir_list[:1]:
		call_simulator(os.path.join(sim_path, benchmark), sim_run_path)

	return


# Generate the sw_connections.txt file require for the simulator
def generate_sw_connections(node_connection_list, path):
	sw_matrix = -1*np.ones((C.SW_MAT_DIM, C.SW_MAT_DIM))
	# Generate the router to core and core to router connection
	for i in range(C.NUM_CORES):
		sw_matrix[i, C.NUM_CORES+i] = 2
		sw_matrix[C.NUM_CORES+i, i] = 2
	# Set the core connection information based on the node connection list
	for node_connection in node_connection_list:
		[start, end] = node_connection
		start_index = get_node_index(start)
		end_index = get_node_index(end)
		sw_matrix[C.NUM_CORES+start_index, C.NUM_CORES+end_index] = 1
		sw_matrix[C.NUM_CORES+end_index, C.NUM_CORES+start_index] = 1
	np.savetxt(path, sw_matrix, fmt='%1d', delimiter='\t', newline='\n', header='', footer='', comments='# ')

def get_node_connection_list(sw_connection_path):
	sw_connection = pd.read_csv(sw_connection_path, delim_whitespace=True, header=None).as_matrix()
	node_connection_list = []
	for i in range(C.NUM_CORES):
		for j in range(i+1, C.NUM_CORES):
			if(sw_connection[C.NUM_CORES+i, C.NUM_CORES+j] == 1):
				node_connection_list.append((U.get_node_position(i), U.get_node_position(j)))
				# print((i,j), (get_node_position(i), get_node_position(j)))
			if(sw_connection[C.NUM_CORES+i, C.NUM_CORES+j] != sw_connection[C.NUM_CORES+j, C.NUM_CORES+i]):
				print("Matrix not symetric ", i, j)

	# connection_idx_list = get_connection_idx_list(connection_list)
	G=nx.Graph()
	G.add_edges_from(node_connection_list)
	path = nx.all_pairs_shortest_path_length(G)
	conn_comp = nx.connected_components(G)
	for conn in conn_comp:
		print("Set ", conn)
	for start_core in G.nodes():
		for end_core in G.nodes():
			if(start_core != end_core):
				# print("Sim", start_core, end_core)
				path_cost = path[start_core][end_core]
				# print(path_cost)


	print("Num router connections ", len(node_connection_list))
	return (node_connection_list)


# Using a dummy simulator for evaluating the cost without calling the actual simulator
class Dummy_Simulator(object):
	def __init__(self, benchmark_path):
		#self.benchmark_path = benchmark_path
		benchmark_path = os.path.join(benchmark_path, "input_benchmark.txt")
		# print(benchmark_path)

		self.comm_freq = self.get_communication_frequency(benchmark_path)
		self.ref_mesh = self.run_mesh()


		return
	def get_communication_frequency(self, path):
		# comm_freq = np.zeros((num_cores, num_cores))
		return pd.read_csv(path, delim_whitespace=True, header=None).as_matrix()

	def _run(self, connection_idx_list):
		# connection_idx_list = get_connection_idx_list(connection_list)
		G=nx.Graph()
		G.add_edges_from(connection_idx_list)
		path = nx.all_pairs_shortest_path_length(G)
		# print(G.edges())
		# print(G.nodes())

		communication_cost = 0
		R = 3
		# Get the objective function
		for start_core in G.nodes():
			for end_core in G.nodes():
				if(start_core != end_core):
					distance = U.get_3d_core_distance(
						U.get_node_position(start_core), 
						U.get_node_position(end_core))
					# print("Sim", start_core, end_core)
					path_cost = path[start_core][end_core]
					comm_cost = self.comm_freq[start_core,end_core]
					# print(U.get_node_position(start_core), \
					# 	U.get_node_position(end_core), \
					# 	distance, path_cost, comm_cost)
					communication_cost += (R*path_cost + distance)*comm_cost

		# print("Communication Frequency ", communication_cost)
		return communication_cost
		# return communication_cost

	def run_mesh(self):
		connection_idx_list = U.generate_mesh_link_list()

		return self._run(connection_idx_list)

	def run(self, feature_vector):
		connection_idx_list = ld.generate_conn_idx_list(feature_vector)

		return (self.ref_mesh - self._run(connection_idx_list))*100/self.ref_mesh




if __name__=="__main__":

	# BASE_FOLDER_RUNNING = "../../../../../data/expensive_experiments/simulator_run/"
	# BASE_FOLDER_SIMULATOR = "../../../../../reference/expensive_experiments/simulator/NoCsim_coreMapped_16thMay2017/"

	# BASE_FOLDER_RUNNING = "../simulator_run/"
	# BASE_FOLDER_SIMULATOR = "../NoCsim_coreMapped_16thMay2017/"

	# run_all_benchmarks(BASE_FOLDER_SIMULATOR, BASE_FOLDER_RUNNING)
	
	benchmark_list = ['dedup', 'canneal', 'lu', 'fft','radix', 'water', 'vips', 'fluid']
	for benchmark in benchmark_list:
		SIM_BASE_FOLDER = "../../../../../reference/expensive_experiments/simulator/NoCsim_coreMapped_16thMay2017/"
		SIM_BASE_FOLDER += benchmark + "/"
		# sim = Simulator(SIM_BASE_FOLDER, "canneal")
		sim = Dummy_Simulator(SIM_BASE_FOLDER)
		print(benchmark, -1*sim.run_mesh())

	INPUT_BENCHMARK = SIM_BASE_FOLDER+"input_benchmark.txt"
	# print get_communication_frequency(INPUT_BENCHMARK)

	# generate_sw_connections(node_connection_list, SW_CONNECTION_1)
	# feature_vector = generate_feature_list(connection_idx_list)
	# print(dummy_simulator(INPUT_BENCHMARK, feature_vector))
	
	# SIM_BASE_FOLDER = "../../../../../reference/expensive_experiments/simulator/NoCsim_coreMapped_16thMay2017/canneal/"

	# INPUT_BENCHMARK = SIM_BASE_FOLDER+"input_benchmark.txt"

	# SW_CONNECTION_1 = SIM_BASE_FOLDER+"sw_connection_1.txt"
	# SW_CONNECTION = SIM_BASE_FOLDER+"sw_connection.txt"

	# node_connection_list = get_node_connection_list(SW_CONNECTION)
