import numpy as np
import pandas as pd
import constants as C
import utility_functions as U
import random
from collections import defaultdict
import time
import random

# Generate all the possible connection options that are possible
# The order in which they are generated is used for generating feature vectors
def generate_core_connection_options():
	core_pos = U.get_2d_core_pos()
	total_cores = len(core_pos)
	core_connection_options = {}
	for i in range(4):
		core_connection_options[i] = []
	#
	for i in range(total_cores):
		for j in range(i+1,total_cores):
			distance = int(round(U.get_core_distance(core_pos[i], core_pos[j])))
			start_core = U.get_node_index((0, core_pos[i][1], core_pos[i][0]))
			end_core = U.get_node_index((0, core_pos[j][1], core_pos[j][0]))
			# Store the distance, start and end core idx
			# core_connection_options.append((start_core, end_core, distance))
			core_connection_options[distance-1].append((start_core, end_core))
			
	# print("Length of connection option ", len(core_connection_options))
	# for i in range(4):
	# 	print(len(core_connection_options[i]))

	#
	core_connection_ordering = []
	for i in range(4):
		for conn in core_connection_options[i]:
			(start_core, end_core) = conn
			core_connection_ordering.append((start_core, end_core, i))

	# (42, 40, 28, 10)
	# print(len(core_connection_options[0]), len(core_connection_options[1]),
	# 	len(core_connection_options[2]), len(core_connection_options[3]))
	# 
	# prob_ord_idx_list = []
	# con_idx = 0
	# for i in range(4):
	# 	for j in range(len(core_connection_options[i])):
	# 		for k in range(C.DISTRIBUTION[i]):
	# 			prob_ord_idx_list.append(con_idx)
	# 		con_idx += 1

	# print(prob_ord_idx_list)



	return core_connection_options, core_connection_ordering

class LinkDistribution(object):
	def __init__(self):
		# Generate all the single link combination
		_, core_connection_ordering = \
			generate_core_connection_options()
		self.core_connection_ordering = core_connection_ordering
		self.total_connections = len(self.core_connection_ordering)
		# self.prob_ord_idx_list = prob_ord_idx_list
		self.vertical_link_list = U.generate_vertical_links_list()
		return
	def generate_conn_idx_list(self, feature_vector):
		core_idx_list = []
		for i in range(len(feature_vector)):
			if(feature_vector[i]):
				lyr_idx = int(i/120)
				core_idx = i - lyr_idx*120
				(start_idx, end_idx, distance) = self.core_connection_ordering[core_idx]
				core_idx_list.append((lyr_idx*16+start_idx, lyr_idx*16+end_idx))
		for conn_idx in self.vertical_link_list:
			core_idx_list.append(conn_idx)

		return core_idx_list

	def reset(self, is_middle_layer):
		self.link_distribution = C.DISTRIBUTION[:]
		self.is_core_available = np.ones(C.NUM_CORE_IN_LAYER)
		self.is_connection_available = np.ones(self.total_connections)
		self.num_core_connected = (C.K_MAX-1-is_middle_layer)*np.ones(C.NUM_CORE_IN_LAYER)
		return

	def check_connection_adds_new_node(self,conn_opt):
		return (self.is_core_available[conn_opt[0]] != self.is_core_available[conn_opt[1]])

	def check_graph_complete(self):
		# print(np.sum(self.is_core_connected), np.sum(self.link_distribution))
		return (not (np.sum(self.is_core_available) == 0 
			and np.sum(self.link_distribution) == 0))

	def is_valid_connection(self, conn_opt, conn_idx):
		(start_idx, end_idx, distance) = conn_opt
		
		return (self.link_distribution[distance]
			and self.is_connection_available[conn_idx]
			and self.num_core_connected[start_idx]
			and self.num_core_connected[end_idx])
	# Iterate over all the connections and identify the conection which has one node in graph and another outside the graph
	# Also make sure the connection is valid
	# Collect all such nodes and randomly choose one connection
	# Repat this until SW conditions and all the nodes are connected
	def get_next_random_connection(self):
		conn_opts_idx = []
		# If all cores are not connected connect them first
		if(np.sum(self.is_core_available) != 0):
			for i in range(self.total_connections):
				conn_opt = self.core_connection_ordering[i]
				# print(conn_opt, 
				# 	check_connection_adds_new_node(is_core_connected_list, conn_opt), 
				# 	link_distribution[conn_opt[2]-1])
				if(self.check_connection_adds_new_node(conn_opt) 
				and self.is_valid_connection(conn_opt, i)):
					conn_opts_idx.append(i)
		# Once all the cores are connected iterate till link distribution goes to zero
		elif(np.sum(self.link_distribution) != 0):
			for i in range(self.total_connections):
				conn_opt = self.core_connection_ordering[i]
				# print(conn_opt, 
				# 	check_connection_adds_new_node(is_core_connected_list, conn_opt), 
				# 	link_distribution[conn_opt[2]-1])
				if(self.is_valid_connection(conn_opt, i)):
					conn_opts_idx.append(i)
		
		# else: return a empty array

		# print("Num conn opts ", len(conn_opts_idx))
		random.shuffle(conn_opts_idx)
		return conn_opts_idx

	def set_flags(self, conn_idx, feature_vector, 
		idx_offset):
		(start_idx, end_idx, distance) = \
			self.core_connection_ordering[conn_idx]
		self.is_core_available[start_idx] = 0
		self.is_core_available[end_idx] = 0
		self.link_distribution[distance] -= 1
		self.is_connection_available[conn_idx] = 0
		self.num_core_connected[start_idx] -= 1
		self.num_core_connected[end_idx] -= 1
		feature_vector[idx_offset+conn_idx] = 1
		# conn_idx_list.append((lyr_offset+start_idx, lyr_offset+end_idx))
		return

	def gen_random_feature(self):
		feature_vector = np.zeros(4*self.total_connections)

		for i in range(4):
			idx_offset = i*self.total_connections
			if(i>0 and i<3):
				self.reset(1)
			else:
				self.reset(0)
			self.set_flags(random.randint(0,self.total_connections-1), \
				feature_vector, idx_offset)

			while(self.check_graph_complete()):
				conn_opts_idx = self.get_next_random_connection()
				# # num_rand_idx_list = [np.sum(self.is_core_available),
				# # 	np.sum(self.link_distribution), len(conn_opts_idx)]
				# # num_rand_idx = int(min(i for i in num_rand_idx_list if i > 0))
				
				# print(num_rand_idx, len(conn_opts_idx),
				# 	np.sum(self.link_distribution), np.sum(self.is_core_available)	)
				conn_idx = conn_opts_idx[0]
				# print(conn_idx, self.link_distribution,
				# 	# self.is_core_available, \
				# 	self.core_connection_ordering[conn_idx])
				
				# conn_idx = conn_opts_idx[i]
				self.set_flags(conn_idx, feature_vector, idx_offset)
		# print(feature_vector)
		# print(conn_idx_list)
				
				
		return feature_vector
	def generate_n_random_feature(self, num_rnds):
		feature_vector_list = []
		for i in range(num_rnds):
			feature_vector_list.append(self.gen_random_feature())
		return np.array(feature_vector_list)

	def generate_successor_graph(self, feature_vector, length_list):
		# find the 
		len_conn_list = []
		len_unconn_list = []
		conn_list = []
		# Initialise a empty array
		for i in range(U.GET_MAX_DISTANCE()):
			len_conn_list.append([])
			len_unconn_list.append([])
		# based on feature vector bucket connection based on length
		for i in range(120):
			(start_idx, end_idx, length) = \
				self.core_connection_ordering[i]
			for j in range(4):
				idx = 120*j+i
				if(feature_vector[idx]):
					len_conn_list[length].append((j,i))
					conn_list.append((j,i))
				else:
					len_unconn_list[length].append((j,i))

		# print("num connection ", len(conn_list), len(len_unconn_list[0]),
		# 	len(len_unconn_list[1]), len(len_unconn_list[2]), len(len_unconn_list[3]))
		successor_graphs = []
		# generate for each length requested
		for length in length_list:
			num_options = 0
			num_skipped = 0

			for conn_info in len_conn_list[length]:
				feature_vector_cpy = np.zeros(4*self.total_connections)
				(lyr_idx, conn_idx) = conn_info
				idx_offset = lyr_idx*self.total_connections
				
				if(lyr_idx>0 and lyr_idx<3):
					self.reset(1)
				else:
					self.reset(0)

				# Initialise the flag with all the connections except the one selected
				for conn_info_1 in conn_list:
					(lyr_idx_1, conn_idx_1) = conn_info_1
					if((lyr_idx == lyr_idx_1) and (conn_idx != conn_idx_1)):
						self.set_flags(conn_idx_1, feature_vector_cpy, idx_offset)
					elif(lyr_idx != lyr_idx_1):
						feature_vector_cpy[lyr_idx_1*self.total_connections+conn_idx_1] = 1
				total_core_available = np.sum(self.is_core_available)
				# print("Validate ", np.sum(feature_vector_cpy - feature_vector))
				
				# For each connection in the unconnected list check whether the graph is connected
				for unconn_info in len_unconn_list[length]:
					(lyr_unconn_idx, unconn_idx) = unconn_info

					# If the layer idx do not match ignore
					if(lyr_idx != lyr_unconn_idx):
						continue

					# If the connection is not available just ignore
					if(not self.is_connection_available[unconn_idx]):
						continue

					num_options += 1

					# print(np.sum(self.is_core_available), 
					# 	self.num_core_connected[start_idx],
					# 	self.num_core_connected[end_idx])

					(uc_start_idx, uc_end_idx, uc_distance) = \
						self.core_connection_ordering[unconn_idx]

					if( total_core_available == (
						self.is_core_available[uc_start_idx] + 
						self.is_core_available[uc_end_idx]) 
					and self.num_core_connected[uc_start_idx] >= 1
					and self.num_core_connected[uc_end_idx] >= 1):
						feature_vector_cpy[lyr_unconn_idx*self.total_connections+unconn_idx] = 1
						successor_graphs.append(np.copy(feature_vector_cpy))
						# if(np.sum(feature_vector_cpy) != 96):
						# 	print("Feature vector does not match ", 
						# 		np.sum(feature_vector_cpy), 
						# 		np.sum(feature_vector))
						feature_vector_cpy[lyr_unconn_idx*self.total_connections+unconn_idx] = 0
					else:
						num_skipped += 1


			# print("Num options ", num_options, num_skipped, length)

# ('Num Successors ', 1331)
# ('Num Successors ', 535)
# ('Num Successors ', 190)
# ('Num Successors ', 34)
# ('Num Successors ', 2090)
# ('Num Successors ', 1384)
# ('Num Successors ', 556)
# ('Num Successors ', 176)
# ('Num Successors ', 36)
# ('Num Successors ', 2152)
# Num Skipped due to all cores not connected and max num connections seems to match the rest
# Ideal num successors:
# 	Connection Distribution: (42, 40, 28, 10)
#	SW Distribution:  [ 16,  5,   2,   1]
#	So 4*16*26 (1664)+5*35*4 (700) + 2*26*4 (208) + 1*9*4(36)
# 	Total = 2608


		# print("Num Successors ", len(successor_graphs))
		return np.asarray(successor_graphs)

# Check for SW values or constraint???
# The init values just using the SW constraint should itself give a good value???
# Issue with model??
# - Replace RF model with GP

def validate_sw_distribution(feature_vector):
	_, core_connection_ordering = generate_core_connection_options()
	connection_list_list = U.generate_conn_idx_list_list(feature_vector)
	ref_dist = [64, 20, 8, 4]
	i = 0
	for connection_list in connection_list_list:
		dist_hist = []
		for idx in range(4):
			dist_hist.append(0)
		for conn in connection_list:
			conn = conn%120
			x,y,dist = core_connection_ordering[conn]
			dist_hist[dist] += 1

		if(dist_hist != ref_dist):
			print("Error ",i, dist_hist)
		# print(dist_hist)
		i+=1
	print("Done validation all checks")
	return


if __name__=="__main__":
	core_connection_options = \
		generate_core_connection_options()
	# print(core_connection_options)
	# print(len(core_connection_options))
	start_time = time.clock()
	feature_vector_list = \
		LinkDistribution().generate_n_random_feature(10)

	print("time taken ", time.clock() - start_time)

	ld = LinkDistribution()
	feature_vector = ld.gen_random_feature()
	conn_idx_list = ld.generate_conn_idx_list(feature_vector)

	successor_graphs = ld.generate_successor_graph(feature_vector,[0])
	successor_graphs = ld.generate_successor_graph(feature_vector,[1])
	successor_graphs = ld.generate_successor_graph(feature_vector,[2])
	successor_graphs = ld.generate_successor_graph(feature_vector,[3])

	successor_graphs = ld.generate_successor_graph(feature_vector,[0,1,2,3])
	# print(feature_vector_list)
	
