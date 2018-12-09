#https://github.com/aryandeshwal/NeuralCompression
#https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark
#/home/aryan/Code/tensorflow --- run tflite models inside this directory 

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

path = '/home/aryan/Downloads/squeezenet_2018_04_27/nasnet_mobile.pb'
path = '/mnt/additional/nitthilan/ml_tutorials/conv/saved_keras_models/SqueezeNet_cifar10_false_1.pb'

with tf.Session() as sess:
    with gfile.FastGFile(path,'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    '''
    graph_nodes=[n for n in graph_def.node]
    names = []
    for t in graph_nodes:
       names.append(t.name)
    print(names)
    '''
    # Print the name of operations in the session
    for op in sess.graph.get_operations():
        print("Operation Name :",op.name)            # Operation nam
        print("Tensor Stats :",str(op.values()))     # Tensor namE
    
    l_input = sess.graph.get_tensor_by_name('input_1:0')
    print(l_input)
    l_output = sess.graph.get_tensor_by_name('final_layer/predictions:0')
    print(l_output)
    
    img = np.zeros((1,48,48,3))
    tf.global_variables_initializer()
    
    Session_out = sess.run( l_output, feed_dict = {l_input : img} )
    print(Session_out)


'''
Tflite commands

bazel build -c opt \
  --config=android_arm \
  --cxxopt='--std=c++11' \
  tensorflow/contrib/lite/tools/benchmark:benchmark_model


adb push bazel-bin/tensorflow/contrib/lite/tools/benchmark/benchmark_model /data/local/tmp

'''
