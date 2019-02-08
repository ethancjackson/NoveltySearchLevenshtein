import json
import numpy as np
from functools import reduce

'''
Represents an individual in the evolutionary search.
'''

class Individual:

    def __init__(self, in_shape, out_shape, init_connect_rate=1.0, init_seed=1, generation_seeds=None):

        # In this version, we will initialize all networks as either
        # - fully connected, or
        # - sparsely connected according to initial random seed and a threshold

        # Input and output shapes
        self.in_shape = in_shape
        self.out_shape = out_shape

        # Connectivity initialization - threshold=1 gives fully connected network
        self.init_connect_rate = init_connect_rate

        # Initial random seed
        self.init_seed = init_seed

        # List of Gaussian seeds
        self.generation_seeds = generation_seeds if generation_seeds is not None else []

        # Neurons - enumeration of [ins, outs, hidden]
        self.in_size = reduce(np.multiply, in_shape)
        self.out_size = reduce(np.multiply, out_shape)
        self.num_neurons = self.in_size + self.out_size

    def copy(self):
        return Individual(in_shape=self.in_shape, out_shape=self.out_shape, init_connect_rate=self.init_connect_rate,
                          init_seed=self.init_seed, generation_seeds=self.generation_seeds.copy())

    def __eq__(self, other):
        if other is None:
            return False
        if len(self.generation_seeds) != len(other.generation_seeds):
            return False
        s1 = np.array([self.init_seed] + self.generation_seeds)
        s2 = np.array(np.zeros_like(s1))
        if not other == None:
            s2 = np.array([other.init_seed] + other.generation_seeds)
        i = 0
        while i < len(s1):
            if np.abs(s1[i] - s2[i]) > 0.001:
                return False
            i += 1
        return True

    def to_JSON(self):
        '''
        Encode this individual as a JSON string.
        :return:
        '''
        json_dict = dict()
        for attr in self.__dict__:
            v = self.__dict__[attr]
            if type(v) == list:
                ls = []
                for i in range(len(v)):
                    ls.append(str(v[i]))
                json_dict[attr] = ls
            else:
                json_dict[attr] = str(self.__dict__[attr])
        return json.dumps(json_dict)

def from_JSON(json_string):
    '''
    Instantiate Individual from JSON string.
    :param json_string:
    :return: Individual
    '''
    ind_attrs = json.loads(json_string)
    ind_params = dict()
    ind_params['in_shape'] = tuple([int(x) for x in ind_attrs['in_shape'].strip('() ').strip(',').split(',')])
    ind_params['out_shape'] = tuple([int(x) for x in ind_attrs['out_shape'].strip('() ').strip(',').split(',')])
    ind_params['init_connect_rate'] = float(ind_attrs['init_connect_rate'])
    ind_params['init_seed'] = int(ind_attrs['init_seed'])
    # ind_params['in_size'] = int(ind_attrs['in_size'])
    # ind_params['out_size'] = int(ind_attrs['out_size'])
    # ind_params['num_neurons'] = int(ind_attrs['num_neurons'])
    # ind_params['arch_ops'] = ind_attrs['arch_ops']
    ind_params['generation_seeds'] = [int(x) for x in ind_attrs['generation_seeds']]
    # ind_params['hid_size'] = int(ind_attrs['hid_size'])
    return Individual(**ind_params)