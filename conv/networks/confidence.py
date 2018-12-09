import numpy as np
import matplotlib as mpl
mpl.use('Agg')

#https://stackoverflow.com/questions/19261077/no-module-named-scipy-stats-why-despite-scipy-being-installed
import scipy
import scipy.stats
from scipy.stats import entropy
import matplotlib.pyplot as plt


def get_conf_sel_list(prediction, threshold, conf_type):
  if(conf_type=="max_prob"):
    conf_list = np.max(prediction, axis=1)
  elif(conf_type=="top_pred_diff"):
    sort_pred = np.sort(prediction, axis=1)
    conf_list = sort_pred[:,-1] - sort_pred[:, -2]
  else:#conf_type="entropy"
    conf_list = np.asarray([1-(entropy(pred)/2.4) for pred in prediction])

  conf_sel_list = conf_list >= threshold

  # print("Conf Type ", conf_type)
  # conf_sel_list_1 = conf_list < threshold
  # output_val = prediction[conf_sel_list_1]
  # print("Confidence ", conf_list.shape, conf_sel_list.shape, threshold)
  # print("Conf list ", prediction[conf_sel_list_1].shape)
  # print("Conf list 1", conf_list[conf_sel_list_1])
  # print("Finding negative value ", output_val[0])
  return conf_sel_list


def get_prob_based_confidence(prediction, true_values, \
  threshold, conf_type):

  conf_sel_list = get_conf_sel_list(prediction, 
    threshold, conf_type)
  total_pred = np.sum(conf_sel_list)

  pred_arg_max = np.argmax(prediction, axis=1)
  pred_arg_max = pred_arg_max[conf_sel_list]
  true_arg_max = np.argmax(true_values, axis=1)
  true_arg_max = true_arg_max[conf_sel_list]
  # print(pred_arg_max.shape, true_arg_max.shape)
  accuracy = np.sum(pred_arg_max == true_arg_max)
  
  # print(pred_max[:10], pred_arg_max[:10], true_arg_max[:10])
  # print(accuracy, total_pred)
  return (accuracy, total_pred, conf_sel_list)

def get_for_all_confidence(prediction, true_values, conf_type):
  # print(prediction.shape, true_values.shape)
  accuracy_list = []
  total_pred_list = []
  conf_threshold = np.asarray(range(10))*0.1
  for conf in conf_threshold:#[0.6]:#
    (accuracy, total_pred, conf_sel_list) = \
    	get_prob_based_confidence(prediction, \
      true_values, conf, conf_type)
    accuracy_list.append(accuracy)
    total_pred_list.append(total_pred)
  return conf_threshold, np.asarray(accuracy_list), np.asarray(total_pred_list) 

def plot_confidence(conf_threshold, accuracy_list, total_pred_list,
  filename):
  fig, ax = plt.subplots()
  width = 0.03 
  plot_total_pred = ax.bar(conf_threshold, total_pred_list,\
   width, color='r', label="Num Predictions")
  plot_accuracy = ax.bar(conf_threshold+width, accuracy_list,\
   width, color='y', label="Num Correct Pred")
  plt.legend(loc=1, borderaxespad=0.)

  fig.savefig(filename)
  return
