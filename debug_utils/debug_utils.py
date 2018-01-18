

import numpy as np
from keras import backend as K

def dump_weights(model):
	# Dump the mean weight and std deviation
    layer_idx = 0
    for layer in model.layers:
        if(len(layer.weights)==2):
            weights = layer.get_weights()
            weight_vector = weights[0]
            bias_vector = weights[1]
            # print("weights ",layer.name, \
            #     weight_vector.shape, bias_vector.shape)
            print(layer.name, "wgt ", np.mean(weight_vector), \
                np.std(weight_vector),\
                "bias ", np.mean(bias_vector), np.std(bias_vector),)
        layer_idx += 1

def dump_layer_details(model):
    # with a Sequential model
    layer_idx = 0
    for layer in model.layers:
        print(layer_idx, layer.name, \
            layer.input_shape, layer.output_shape)
        # if(len(layer.weights)):
        #     print("weights ",layer.get_weights())
        # print("updates ",layer.get_updates_for(0))
        layer_idx += 1
    return

def dump_gradients(model, input_data):
    # https://github.com/keras-team/keras/issues/2226 - How to get gradients
    weights = model.trainable_weights # weight tensors
    weights = [layer.get_weights() for layer in model.layers if layer.trainable] # filter down weights tensors to only ones which are trainable
    gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors

    print(model.total_loss, len(weights))
    print(gradients)

    input_tensors = [model.inputs[0], # input data
                 model.sample_weights[0], # how much to weight each sample by
                 model.targets[0], # labels
                 K.learning_phase(), # train or test mode
                ]

    get_gradients = K.function(inputs=input_tensors, outputs=gradients)

    # inputs = [x_test, # X
    #       [1], # sample weights
    #       x_test, # y
    #       0 # learning phase in TEST mode
    # ]

    for (weight_name, gradients) in zip(weights, get_gradients([input_data])):
        print(weight_name.name, np.mean(gradients), np.std(gradients))

    return

def get_intermediate_output(model, start_layer_id,
    end_layer_id, input_data):
    # print(dir(model.layers[start_layer_id]))

    intermediate_function = K.function([model.layers[start_layer_id].input, 
        K.learning_phase()],
        [model.layers[end_layer_id].output])

    intermediate_output = intermediate_function([input_data, 0])[0]
    return intermediate_output


def get_callbacks():
    tensorboard = TensorBoard(log_dir='./logs/vae_'+str(num_epocs), 
        histogram_freq=1, batch_size=32, 
        write_graph=False, write_grads=True, write_images=False, 
        embeddings_freq=0, embeddings_layer_names=None, 
        embeddings_metadata=None)


    modelCheckpoint = ModelCheckpoint(data_base_folder+"weights.{epoch:02d}-{val_loss:.2f}.hdf5", 
        monitor='loss', verbose=0, save_best_only=True, 
        save_weights_only=True, mode='auto', period=1)

    callbacks = [
                # modelCheckpoint
                #   earlyStopping, 
                #   reduceonplateau,
                #   csv_logger
                ]
    return callbacks

