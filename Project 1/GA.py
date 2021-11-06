import numpy as np
import pandas as pd
import random
import math

numberOfCity = 10

city_10 = pd.DataFrame(np.array([[0.3642,0.7779], [0.7185,0.8312], [0.0986,0.5891], [0.2954,0.9606],
                                 [0.5951,0.4647], [0.6697,0.7657], [0.4353, 0.1709], [0.2131,0.8349],
                                 [0.3479,0.6984], [0.4516,0.0488]]),
                       columns=['x', 'y'])


def init_para(pop_size=10, mutation_rate=0.03, crossover_rate=0.8, chromosome_len=30, max_generation=50):
    """
    Initialize parameters for Genetic algorithms.
    crossover_rate: Probability of crossover (typically near 1)
    mutation_rate: Probability of mutation (typically <.1)
    :return: A dictionary with parameters as key
    """
    para_dict = {'pop_size': pop_size, 'mutation_rate': mutation_rate, 'crossover_rate': crossover_rate,
                 'chromosome_len': chromosome_len, 'max_generation': max_generation}
    return para_dict


def generate_chromosome(numberOfCity):
    """
    generate a chromosome based on number of city. The chromosome has first and end value the same for indicating
    circular motion.
    :param numberOfCity: number of cities need to be traveled
    :return: List chromosome
    """
    chromosome = random.sample(range(0, numberOfCity), numberOfCity)
    chromosome.append(chromosome[0])
    return chromosome


def cal_fitness(chromosome, city_table):
    """
    This function calculate the chromosome's fitness score which equal to average of sum of distance * 1000.
    """
    fitness = 0
    for i in range(len(chromosome)-1):
        city_a_x = city_table.iloc[chromosome[i], 0]
        city_a_y = city_table.iloc[chromosome[i], 1]
        city_b_x = city_table.iloc[chromosome[i+1], 0]
        city_b_y = city_table.iloc[chromosome[i+1], 1]
        dis = np.round(math.sqrt(((city_a_x - city_b_x)**2) + ((city_a_y - city_b_y)**2)), 5)
        fitness += dis
    fitness = fitness/(len(chromosome)-1)
    return (1/fitness) * 100


def init_pop(pop_size, city_table):
    """
    This function is use to initialize the initial evolving population of the GA in a initialized pandas Dataframe
    :return: DataFrame pop_table
    """
    pop_table = pd.DataFrame(columns=['chromosome', 'fitness'], index=range(pop_size))
    for i in range(pop_size):
        chromosome = generate_chromosome(len(city_table))
        pop_table.at[i, 'chromosome'] = chromosome
        pop_table.at[i, 'fitness'] = cal_fitness(chromosome, city_table)
    #print(pop_table)
    return pop_table


def tournament_selection(pop_table, k):
    """
    Select k chromosome from the tournament
    :param pop_table: population city
    :param k: k chromosome are selected in this process. k should be less than number of cities.
    :return: dataframe of one row which is the final selection of one tournament without replacement
    """
    # number_of_city = len(pop_table['chromosome'][0])-1
    k_select = random.sample(range(0, len(pop_table)-1), k)
    # print('k_select is '+ str(k_select))
    k_table = pd.DataFrame(columns=['chromosome', 'fitness'])
    k_final = pd.DataFrame(columns=['chromosome', 'fitness'])
    for i in range(len(k_select)):
        extract = pop_table.iloc[k_select[i]]
        k_table = k_table.append(extract)
    # Select the highest fitness score in the selected k chromosome
    h = k_table['fitness'].idxmax()
    k_final = k_final.append(k_table.loc[h])
    return k_final


def selection(pop_table, k):
    """
    Perform 2 tournament selection and select the best one among them.
    """
    k1 = tournament_selection(pop_table, k)
    k2 = tournament_selection(pop_table, k)
    if k1['fitness'].iloc[0] > k2['fitness'].iloc[0]:
        return k1
    return k2


if __name__ == "__main__":
    pop_table = init_pop(20, city_10)
    parent1 = selection(pop_table, 3)
    parent2 = selection(pop_table, 3)