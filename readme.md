Horizontal Filter Training:
==========================

Procedure:
----------
- Train base network (resize_factor = 0)
	- python3 training.py <network> <data_type>
	- network = conv, vgg, wrn
		- conv - base conv net, vgg - vgg style net, wrn - wide residual net
		- data_type - mnist, cifar10, cifar100
	- eg: python3 training.py conv mnist
	- This dumps the trained model in saved_models directory
- Transfer learning training the other resize factors (resize_factor = 1, 2, 3)
	- python3 transfer_learning.py <network> <data_type> <resize_factor> <weight_path>
	- resize_factor - 1, 2, 3 
		- 0 - the base network (Finest network)
		- 1, 2, 3 - fine to coarsest network
	- weight_path - path to the base pretrained network (resize_factor = 0)
	- eg: python3 transfer_learning.py conv cifar100 1 ../../../data/ml_tutorial/conv/cifar100_vgg_v2/keras_cifar100_weight_0.h5
	- This dumps the trained model in saved_models directory
- Dump the images predicted using the finest to coarsest network for each combination
	- python3 testing.py 
	- Set the save_dir variable to the base folder of all the trained networks
- Run the optimisation to estimate the thresholds for the multiple networks
	- python3 optimal_dist.py
	- Set the following variables
		- save_dir - path for all the trained networks
		- energy_values - energy values of the network measured offline for each network
		- bayes_args - choose two or three threholds based on 3/4 networks