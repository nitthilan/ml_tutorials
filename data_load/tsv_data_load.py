
import numpy as np
import random

DUMP_FILE = "../../data/ml_tutorials/autoencoder/tsv_distribution_100000.npz"

def RGen(nVL, budget):

	randDesign = [0 for i in range(8*nVL)]
	for budgetLeft in range(budget):
		randTSV = random.choice([i for i in range(len(randDesign))])
		randDesign[randTSV] +=1

	return randDesign



def save_data(x):
    np.savez_compressed(DUMP_FILE, x)
    return
    

def return_saved():
    return np.load(DUMP_FILE)['arr_0']

def get_data(is_stored, budget, numDesigns):
    if(not is_stored):
        rand_designs = RGenAlternative(budget, numDesigns)
        print(rand_designs.shape)
        # rand_design_val = dummy_utility(rand_designs)
        save_data(rand_designs)
        # print(len(rand_designs), rand_desing_val)
        return rand_designs
    else:
        rand_designs = return_saved()
        # rand_designs[rand_designs<0.5] = -1
        return rand_designs[:numDesigns]


def RGenAlternative(budget, numDesigns):
	
	vlIndexList = list()

	for i in range(48):
		if i in [k for k in range(16,32)]:
			vlIndexList.append(i)
			vlIndexList.append(i)
		else:
			vlIndexList.append(i)

	tsvIndexList = list()

	for i in range(8):
		if i in [0,2,6]:
			tsvIndexList.append(i)
		elif i in [1,3,5,7]:
			tsvIndexList.append(i)
			tsvIndexList.append(i)
		else:
			for j in range(12):
				tsvIndexList.append(i)

	all_designs = []
	for _id in range(numDesigns):
		design = np.zeros((48, 8))
		for budgetLeft in range(budget):
			i = random.choice(vlIndexList)
			j = random.choice(tsvIndexList)
			design[i, j] = design[i, j] + 1
		design = design.ravel()
		all_designs.append(design)
		# design = list(design)

	return np.array(all_designs)

if __name__=="__main__":
	NUM_DATA = 100000
	data = get_data(False, 19, NUM_DATA)
	print(data.shape)
	print(data.shape)
	print(data[0])
	print(data[9])
	print(np.sum(data, axis=1))
	data = get_data(True, 19, 10)
	print(data.shape)