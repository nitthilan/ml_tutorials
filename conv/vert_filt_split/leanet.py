
import threshold_optimiser as to
import os
import sys
import numpy as np


# cd /usr/local/lib/python2.7/dist-packages/spearmint-0.1-py2.7-linux-x86_64.egg/spearmint/
# sudo python main.py /mnt/additional/nitthilan/ml_tutorials/conv/vert_filt_split/
# sudo python cleanup.py /mnt/additional/nitthilan/ml_tutorials/conv/vert_filt_split/
# /mnt/additional/nitthilan/ml_tutorials/conv/vert_filt_split/


dict_to_conf_map = {
    0:"max_prob", 1:"top_pred_diff", 2:"entropy"
}

# Change this to 13 or 14
config = 1

config_list = to.config_list
BASE_FOLDER = to.BASE_FOLDER
for i in range(len(config_list[config][0])):
    config_list[config][0][i] = os.path.join(BASE_FOLDER, config_list[config][0][i])

# Edit configuration
optimiser_values = config_list[config][1]
model_list = config_list[config][0]
ctype = 0#dict_to_conf_map[0]

print(config_list)
print(BASE_FOLDER)


def main(job_id, params):
    # c1 = params['X0']
    # c2 = params['X1']
    # c3 = params['X2']
    # c4 = params['X3']

    # # print(c1, c2, c3, c4, ctype)

    # optmiser_func = to.Optimiser(optimiser_values, model_list)
    # total_acc_nor, value_spent_nor = \
    #     optmiser_func.find_acc_energy(c1=c1, c2=c2, c3=c3, c4=c4, ctype=ctype)
    # # print(total_acc_nor, value_spent_nor)
    # return {
    #     "Accuracy" : total_acc_nor, 
    #     "Energy" : value_spent_nor, 
    # }
    try:
        c1 = params['X0'][0]
        c2 = params['X1'][0]
        c3 = params['X2'][0]
        c4 = params['X3'][0]

        # print(c1, c2, c3, c4, ctype)

        optmiser_func = to.Optimiser(optimiser_values, model_list, True)
        total_acc_nor, value_spent_nor = \
            optmiser_func.find_acc_energy(c1=c1, c2=c2, c3=c3, c4=c4, ctype=ctype)
        # print(total_acc_nor, value_spent_nor)
        return {
            "Accuracy" : -1*total_acc_nor, 
            "Energy" : -1*value_spent_nor, 
        }
    except Exception as ex:
        print ex
        print 'An error occurred in threshold estimation'
        return np.nan


# print(main(1000, {'X0':0.5, 'X1':0.5, 'X2':0.5, 'X3':0.5 }))