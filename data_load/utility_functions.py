
import constants as C
import numpy as np
def get_distance(x,y,x1,y1):
	return (((x-x1)**2+(y-y1)**2)**0.5)

def get_core_distance(start,end):
	return get_distance(start[0], start[1], end[0], end[1])

def get_3d_distance(x,y,z,x1,y1,z1):
	return (((x-x1)**2 + (y-y1)**2 + (z-z1)**2)**0.5)

def get_3d_core_distance(start, end):
	return get_3d_distance(start[0], start[1], start[2], end[0], end[1], end[2])

def get_num_vertical_link(num_layers, num_cores):
	return (num_layers-1)*int(num_cores/num_layers)

def get_node_position(index):
	x = int(index%C.X_WTH)
	y = int(((index - x)/C.X_WTH)%C.Y_HGT)
	z = int(((index - y*C.Y_HGT - x)/(C.X_WTH*C.Y_HGT)))
	return (z,y,x)

def get_node_index(position):
	z,y,x = position
	return int(z*C.X_WTH*C.Y_HGT + y*C.X_WTH + x)

def GET_MAX_DISTANCE():
	return int(round(get_distance(0,0,C.X_WTH-1,C.Y_HGT-1)))

def get_connection_idx_list(connection_list):
	connection_idx_list = []
	for start,end in connection_list:
		# print(start,end)
		connection_idx_list.append((get_node_index(start), get_node_index(end)))

	return connection_idx_list

def GET_NUM_CONNECTION():
	num_cores_in_plane = C.X_WTH*C.Y_HGT
	num_connection_in_plane = (num_cores_in_plane*(num_cores_in_plane-1))/2
	return int(C.Z_LYR*num_connection_in_plane)

def generate_feature_list(connection_idx_list_list):
	N = len(connection_idx_list_list)
	feature_list = np.zeros((N, GET_NUM_CONNECTION()))
	# print(feature_list.shape)
	for i in range(N):
		connection_idx_list = connection_idx_list_list[i]
		for conn_idx in connection_idx_list:
			feature_list[i][conn_idx] = 1

	return feature_list

def generate_conn_idx_list_list(feature_vector_list):
	connection_idx_list_list = []
	(N, M) = feature_vector_list.shape
	# print("Shape ", N, M)

	for i in range(N):
		connection_idx_list = []
		for j in range(GET_NUM_CONNECTION()):
			if(feature_vector_list[i][j]):
				connection_idx_list.append(j)
		connection_idx_list_list.append(connection_idx_list)

	return connection_idx_list_list

def generate_vertical_links_list():
	vertical_link_list = []
	for i in range(C.Z_LYR-1):
		for j in range(C.Y_HGT):
			for k in range(C.X_WTH):
				vertical_link_list.append( 
					( get_node_index((i,j,k)) , \
						get_node_index((i+1,j,k)) ) )

	# print(vertical_link_list)
	return vertical_link_list

def generate_mesh_link_list():
	mesh_link_list = []
	for i in range(C.Z_LYR-1):
		for j in range(C.Y_HGT):
			for k in range(C.X_WTH):
				mesh_link_list.append( 
					( get_node_index((i,j,k)) , \
						get_node_index((i+1,j,k)) ) )
				mesh_link_list.append( 
					( get_node_index((j,i,k)) , \
						get_node_index((j,i+1,k)) ) )
				mesh_link_list.append( 
					( get_node_index((j,k,i)) , \
						get_node_index((j,k,i+1)) ) )


	# print(mesh_link_list)
	return mesh_link_list

def get_2d_core_pos():
	# Since all the connection is in a plane we can generate the 
	# connection information for a single plane and replicate it for all planes

	# Generate all the (x,y) positions
	core_pos = []
	for i in range(C.Y_HGT):
		for j in range(C.X_WTH):
			core_pos.append((i,j))
	return core_pos




if __name__ == "__main__":
	print get_distance(0, 0, 3, 3)
	print get_num_vertical_link(4, 64)
	print get_core_distance((0,0), (3,3))
	print get_3d_core_distance((0,0,0),(1,2,3))
