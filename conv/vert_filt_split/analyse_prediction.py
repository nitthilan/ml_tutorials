import numpy as np 
# from bayes_opt import BayesianOptimization
import os

import confidence as cf
# import get_data as gd
from PIL import Image
import sys
from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

seed = 7
np.random.seed(seed)

def get_pred_y_test(models_path_list, is_train):
	predict_gen_list = []
	y_test_list = []
	# load all the prediction and true value
	for pred_path in models_path_list:
		if(is_train):
			predict_path = pred_path[:-3]+"_train_predict.npz"
		else:
			predict_path = pred_path[:-3]+"_predict.npz"
		loaded_data = np.load(predict_path)
		predict_gen = loaded_data['arr_0']
		y_test = loaded_data['arr_1']
		# print("Predicted value ", y_test.shape, predict_gen.shape)
		predict_gen_list.append(predict_gen)
		y_test_list.append(y_test)
	return predict_gen_list, y_test_list
def get_accuracy(num_models, ctype, predict_gen_list, y_test_list):
	accuracy_values = {}
	min_acc = 100
	max_acc = 0
	for idx in range(num_models):
		(accuracy, total_pred, conf_sel_list) = \
			cf.get_prob_based_confidence(\
				predict_gen_list[idx], \
				y_test_list[idx], \
	  			0.0, ctype)
		accuracy_values[idx] = accuracy*1.0/total_pred
		# print(accuracy, total_pred, accuracy_values[idx])

		# Min and max accuaracy
		if(min_acc > accuracy_values[idx]):
			min_acc = accuracy_values[idx]
		if(max_acc < accuracy_values[idx]):
			max_acc = accuracy_values[idx]
	return accuracy_values, min_acc, max_acc

load_conv_cifar10_list = [
	"./saved_keras_models/conv_cifar10_false_4.h5",
	"./saved_models_bkup/vert_filt_conv_cifar10_3.h5",
	"./saved_models_bkup/vert_filt_conv_cifar10_2.h5",	
]
time_conv_cifar10_list = [3.373, 2.02, 1.0]#[36854, 22168, 10926]#[74, 56, 35]

predict_train_gen_list, y_train_list = get_pred_y_test(load_conv_cifar10_list,\
	 True)

accuracy_train_values, min_acc, max_acc = get_accuracy(3, 0, predict_train_gen_list,
	y_train_list)


predict_test_gen_list, y_test_list = get_pred_y_test(load_conv_cifar10_list,\
	 False)

accuracy_test_values, min_acc, max_acc = get_accuracy(3, 0, predict_test_gen_list,
	y_test_list)

# print(accuracy_train_values, accuracy_test_values)

# print(predict_gen_list[0][:10], y_test_list[0][:10])

pred_arg_max = np.argmax(predict_train_gen_list[0], axis=1)
y_arg_max = np.argmax(y_train_list[0], axis=1)

final_match = pred_arg_max == y_arg_max
# print(pred_arg_max[:10], y_arg_max[:10], np.sum((pred_arg_max == y_arg_max)))

pred_arg_max = np.argmax(predict_train_gen_list[2], axis=1)
y_arg_max = np.argmax(y_train_list[2], axis=1)
final_match_1 = pred_arg_max == y_arg_max



pred_test_arg_max = np.argmax(predict_test_gen_list[2], axis=1)
y_test_arg_max = np.argmax(y_test_list[2], axis=1)
final_match_test = pred_test_arg_max == y_test_arg_max
predict_test_gen_list, y_test_list

encoder = LabelEncoder()
encoder.fit(final_match_1)
encoded_Y = encoder.transform(final_match_1)
print(encoded_Y[:10])

def create_baseline():
	# create model
	model = Sequential()
	# model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

	model.add(Dense(1, input_dim=10, kernel_initializer='normal', activation='sigmoid'))

	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# estimator = KerasClassifier(build_fn=create_baseline, epochs=50, batch_size=128, verbose=1)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, predict_train_gen_list[2], \
# 	encoded_Y, cv=kfold)
# print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

x_train = predict_train_gen_list[2]
y_train = final_match_1

x_test = predict_test_gen_list[2]
y_test = final_match_test

model = create_baseline()
model.fit(x_train, y_train,
	        epochs=200, batch_size=128, verbose=1,
	        validation_data=(x_test, y_test),
	        shuffle=True)


print(np.sum(final_match), np.sum(final_match_1), np.sum(final_match[final_match_1]))
