import numpy as np
import pandas as pd
import random
import math
import heapq
import matplotlib.pyplot as plt
import GA as g
import os

# print(os.getcwd())
# par = g.init_para(pop_size=100, max_gen=300, k_tournament=10, crossover_rate=0.95, mutation_rate=0.03)
# result = g.GA_process(par)


def fixed_pos(number, chromosome):
    return


# def init_pop(pop_size, city_table, fixed_pos=None):
#     """
#     This function is use to initialize the initial evolving population of the GA in a initialized pandas Dataframe
#     :return: DataFrame pop_table
#     """
#     pop_table = pd.DataFrame(columns=['chromosome', 'fitness'], index=range(pop_size))
#     for i in range(pop_size):
#         chromosome = generate_chromosome(len(city_table))
#         pop_table.at[i, 'chromosome'] = chromosome
#         pop_table.at[i, 'fitness'] = cal_fitness(chromosome, city_table)
#     #print(pop_table)
#     return pop_table


def generate_chromosome(numberOfCity, fixed_pos=[]):
    """
    generate a chromosome based on number of city. The chromosome has first and end value the same for indicating
    circular motion.
    :param numberOfCity: number of cities need to be traveled
    :param fixed_pos: List of position that need to be traversed sequentially
    :return: List chromosome
    """
    chromosome = random.sample(range(0, numberOfCity), numberOfCity)
    remain = [x for x in chromosome if x not in fixed_pos]
    final = fixed_pos + remain
    return final

b=generate_chromosome(10)
print(b)