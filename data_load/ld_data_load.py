import os
import os, sys

import numpy as np

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import link_distribution_16 as ld16
ld = ld16.LinkDistribution()
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import bayesian_helpers as bh

# DUMP_FILE = "design_data_latent.pickle"
DUMP_FILE = "../../data/ml_tutorials/autoencoder/link_distribution_100000.npz"


def save_data(x):

    # np.savez(DUMP_FILE, x=x)
    np.savez_compressed(DUMP_FILE, x)
    
    # # Saving the objects:
    # with open(DUMP_FILE, 'w') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump(x, f)

    # return
def return_saved():
    # print(np.load(DUMP_FILE).keys())
    return np.load(DUMP_FILE)['arr_0']
    
    # # Saving the objects:
    # with open(DUMP_FILE, 'r') as f:  # Python 3: open(..., 'wb')
    #     return pickle.load(f)

def get_data(is_stored, num_data):
    if(not is_stored):
        rand_designs = ld.generate_n_random_feature(num_data)
        print(rand_designs.shape)
        # rand_design_val = dummy_utility(rand_designs)
        save_data(rand_designs)
        # print(len(rand_designs), rand_desing_val)
        return rand_designs
    else:
        rand_designs = return_saved()
        print(rand_designs.shape)
        # rand_designs[rand_designs<0.5] = -1
        return rand_designs[:num_data]

def split_vector(d):
    num_vector = d.shape[0]
    print(num_vector)
    vector_array = np.zeros((4*num_vector, 120))
    print(vector_array.shape)
    for i in range(num_vector):
        for j in range(4):
            vector_array[4*i+j] = d[i,j*120:(j+1)*120]
    
    # num_unique = bh.unique_rows(vector_array)
    # print(np.sum(num_unique))
    return vector_array

def get_adj_matrix(d):
    num_vector = d.shape[0]
    _, core_connection_ordering = \
            ld16.generate_core_connection_options()
    print(num_vector)
    adj_array = np.zeros((4*num_vector, 16, 16))
    print(adj_array.shape)
    for i in range(num_vector):
        for j in range(4):
            for k in range(120):
                if(d[i,j*120+k]):
                    (start_idx, end_idx, dist) = core_connection_ordering[k]
                    adj_array[i*4+j, start_idx, end_idx] = 1
                    adj_array[i*4+j, end_idx, start_idx] = 1
    
    # num_unique = bh.unique_rows(vector_array)
    # print(np.sum(num_unique))
    return adj_array

def get_data_adj_mat():
    data = get_data(True)
    print(data.shape)
    # split_vector(data)
    adj_mat = get_adj_matrix(data)
    return adj_mat

if __name__=="__main__":
    NUM_DATA = 100000
    data = get_data(False, NUM_DATA)
    print(data.shape)
    data = get_data(True, 10)
    print(data.shape)