import numpy as np
import pandas as pd
import random
import math

numberOfCity = 10
pop_size = 10


def init_pop(pop_size, numberOfCity):
    """
    This function is use to initialize the initial evolving population of the GA in a initialized pandas Dataframe
    :return: DataFrame pop_table
    """
    pop_table = pd.DataFrame(columns=['chromosome', 'fitness'], index=range(pop_size))
    for i in range(pop_size):
        pop_table.at[i, 'chromosome'] = generate_chromosome(numberOfCity)
    return pop_table

def generate_chromosome(numberOfCity):
    chromosome = random.sample(range(0, numberOfCity), numberOfCity)
    chromosome.append(chromosome[0])
    return chromosome

a = init_pop(10, 5)
print(a)