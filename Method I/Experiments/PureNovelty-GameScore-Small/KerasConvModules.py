from keras.layers import Conv2D, Dense, Flatten
from keras import Sequential
from keras.initializers import glorot_normal

def gen_DQN_architecture(input_shape, output_size, seed):
    conv_layer1 = Conv2D(32, 8, strides=4, use_bias=False, kernel_initializer=glorot_normal(seed), activation='relu', input_shape=input_shape)
    conv_layer2 = Conv2D(64, 4, strides=2, use_bias=False, kernel_initializer=glorot_normal(seed+1), activation='relu')
    conv_layer3 = Conv2D(64, 3, strides=1, use_bias=False, kernel_initializer=glorot_normal(seed+2), activation='relu')
    dense_layer = Dense(512, use_bias=False, kernel_initializer=glorot_normal(seed+3), activation='relu')
    flatten_layer = Flatten()
    output_layer = Dense(output_size, use_bias=False, kernel_initializer=glorot_normal(seed+4), activation='relu')

    return Sequential([conv_layer1, conv_layer2, conv_layer3, dense_layer, flatten_layer, output_layer])
