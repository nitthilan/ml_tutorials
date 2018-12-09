
# Referenced from
# https://github.com/rcmalli/keras-squeezenet/blob/master/keras_squeezenet/squeezenet.py
from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.utils import get_file
from keras.utils import layer_utils
from keras.layers import Dense

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

WEIGHTS_PATH = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_NO_TOP = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Modular function for Fire Node

def fire_module(x, fire_id, squeeze=16, expand=64, use_bias=True):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    
    x = Convolution2D(squeeze, (1, 1), padding='valid', use_bias=use_bias, name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', use_bias=use_bias, name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', use_bias=use_bias, name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x


# Original SqueezeNet from paper.

def SqueezeNet(include_top=True, weights='imagenet',
               input_tensor=None, input_shape=None,
               pooling=None, classes=1000,
               num_filter=1, use_bias=True):
    """Instantiates the SqueezeNet architecture.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')


    input_shape = _obtain_input_shape(input_shape,
                                      default_size=227,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if(num_filter == 1):
        F1=64; FS2=16; FE2=64; FS3=32; FE3=128; FS4=48; FE4=192; FS5=64; FE5=256;
    elif(num_filter == 2):
        F1=56; FS2=14; FE2=56; FS3=28; FE3=112; FS4=42; FE4=168; FS5=56; FE5=224;
    elif(num_filter == 3):
        F1=48; FS2=12; FE2=48; FS3=24; FE3= 96; FS4=36; FE4=144; FS5=48; FE5=192;
    elif(num_filter == 4):
        F1=32; FS2= 8; FE2=32; FS3=16; FE3= 64; FS4=24; FE4= 96; FS5=32; FE5=128;

    x = Convolution2D(F1, (3, 3), strides=(2, 2), 
        padding='valid', name='conv1', use_bias=use_bias)(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=FS2, expand=FE2, use_bias=use_bias)
    x = fire_module(x, fire_id=3, squeeze=FS2, expand=FE2, use_bias=use_bias)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=FS3, expand=FE3, use_bias=use_bias)
    x = fire_module(x, fire_id=5, squeeze=FS3, expand=FE3, use_bias=use_bias)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=FS4, expand=FE4, use_bias=use_bias)
    x = fire_module(x, fire_id=7, squeeze=FS4, expand=FE4, use_bias=use_bias)
    x = fire_module(x, fire_id=8, squeeze=FS5, expand=FE5, use_bias=use_bias)
    x = fire_module(x, fire_id=9, squeeze=FS5, expand=FE5, use_bias=use_bias) 
    
    if include_top:
        # It's not obvious where to cut the network... 
        # Could do the 8th or 9th layer... some work recommends cutting earlier layers.
        x = fire_module(x, fire_id=10, squeeze=FS5, expand=FE5, use_bias=use_bias) 
        x = Dropout(0.5, name='drop9')(x)
        x = Convolution2D(classes, (1, 1), padding='valid', name='conv10', 
            use_bias=use_bias)(x)
        x = Activation('relu', name='relu_conv10')(x)
        # x = GlobalAveragePooling2D()(x)
        x = GlobalMaxPooling2D()(x)
        x = Activation('softmax', name='loss')(x)

        # x = GlobalAveragePooling2D()(x)
        # x = Dense(256, activation='relu')(x)
        # # x = Activation('relu')(x)
        # x = Dropout(0.5)(x)
        # # x = Dense(num_output)(x)
        # # x = Activation('softmax')(x)
        # x = Dense(classes, activation='softmax',
        #                      use_bias=True, name='Logits')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling=='max':
            x = GlobalMaxPooling2D()(x)
        elif pooling==None:
            pass
        else:
            raise ValueError("Unknown argument for 'pooling'=" + pooling)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='squeezenet')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
            
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model
