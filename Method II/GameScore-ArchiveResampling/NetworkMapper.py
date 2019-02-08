from Individual import Individual
from numpy.random import RandomState
from numpy import add, sum, array

"""
Maps an Individual to a Network.
"""

def mutate_conv_module(individual, conv_module, mut_power=0.002):
    """
    Use individual's generation seeds to apply gaussian noise to conv_module.
    :param individual: Individual object used to provide list of generation seeds.
    :param conv_module: Keras model representing a convolutional neural network model.
    """
    old_weights = conv_module.get_weights()
    new_weights = []
    for old_array in old_weights:
        new_array = array(old_array)
        for seed in individual.generation_seeds:
            rnd = RandomState(seed=seed).normal
            add(new_array, rnd(scale=1 * mut_power, size=new_array.shape), out=new_array)
        new_weights.append(new_array)
    conv_module.set_weights(new_weights)

def mutate_conv_module_last_layer_only(individual, conv_module, mut_power=0.002):
    """
    Use individual's generation seeds to apply gaussian noise to conv_module.
    :param individual: Individual object used to provide list of generation seeds.
    :param conv_module: Keras model representing a convolutional neural network model.
    """
    old_weights = conv_module.get_weights()
    new_weights = []
    for layer_index, old_array in enumerate(old_weights):
        new_array = array(old_array)
        # Only update weights in last layer...
        if layer_index == len(old_weights) - 1:
            for seed in individual.generation_seeds:
                rnd = RandomState(seed=seed).normal
                add(new_array, rnd(scale=1 * mut_power, size=new_array.shape), out=new_array)
        new_weights.append(new_array)
    conv_module.set_weights(new_weights)