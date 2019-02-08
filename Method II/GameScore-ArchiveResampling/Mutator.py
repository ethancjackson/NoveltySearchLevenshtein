from Individual import Individual
from numpy.random import randint

"""
A mutator selects and applies mutation operations (architecture and parameter) to an Individual according
to its parameters. In this version, it is important to note that mutations DO NOT consider network topology.
"""

def mutate(individual):
    assert type(individual) == Individual
    individual.generation_seeds.append(randint(low=0, high=4294967265))
