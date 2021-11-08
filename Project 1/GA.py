import numpy as np
import pandas as pd
import random
import math
import heapq
import matplotlib.pyplot as plt
import os
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

numberOfCity = 10

city_10 = pd.DataFrame(np.array([[0.3642,0.7770], [0.7185,0.8312], [0.0986,0.5891], [0.2954,0.9606],
                                 [0.5951,0.4647], [0.6697,0.7657], [0.4353, 0.1709], [0.2131,0.8349],
                                 [0.3479,0.6984], [0.4516,0.0488]]),
                       columns=['x', 'y'])


def init_para(pop_size=10, mutation_rate=0.03, crossover_rate=0.8, chromosome_len=30, k_tournament=4, max_gen=21):
    """
    Initialize parameters for Genetic algorithms.
    crossover_rate: Probability of crossover (typically near 1)
    mutation_rate: Probability of mutation (typically <.1)
    :return: A dictionary with parameters as key
    """
    para_dict = {'pop_size': pop_size, 'mutation_rate': mutation_rate, 'crossover_rate': crossover_rate,
                 'chromosome_len': chromosome_len, 'k_tournament': k_tournament, 'max_generation': max_gen}
    return para_dict


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


def cal_fitness(chromosome, city_table):
    """
    This function calculate the chromosome's fitness score which equal to average of sum of distance * 1000.
    """
    fitness = 0
    for i in range(-1, len(chromosome)-1):
        city_a_x = city_table.iloc[chromosome[i], 0]
        city_a_y = city_table.iloc[chromosome[i], 1]
        city_b_x = city_table.iloc[chromosome[i+1], 0]
        city_b_y = city_table.iloc[chromosome[i+1], 1]
        dis = math.sqrt(((city_a_x - city_b_x)**2) + ((city_a_y - city_b_y)**2))
        fitness += dis
        # np.round((1 / fitness) * 100, 3)
    return (1 / fitness) * 100


def init_pop(pop_size, city_table=city_10, fixed_pos=[]):
    """
    This function is use to initialize the initial evolving population of the GA in a initialized pandas Dataframe
    :return: DataFrame pop_table
    """
    pop_table = pd.DataFrame(columns=['chromosome', 'fitness'], index=range(pop_size))
    for i in range(pop_size):
        chromosome = generate_chromosome(len(city_table), fixed_pos)
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
        extract = pop_table.iloc[k_select[i]].copy()
        k_table = k_table.append(extract)
    # Select the highest fitness score in the selected k chromosome
    h = k_table['fitness'].idxmax()
    k_final = k_final.append(k_table.loc[h])
    return k_final


def selection(pop_table, k):
    """
    Perform 2 tournament selection and make sure they are different.
    """
    k1 = tournament_selection(pop_table, k)
    k2 = tournament_selection(pop_table, k)
    while k1['chromosome'].iloc[0] == k2['chromosome'].iloc[0]:
        k2 = tournament_selection(pop_table, k)
    k1 = k1.append(k2)
    return k1


def crossover(parent1, parent2):
    """
    Perform recombination for order-based representation
    :return: dataframe with 2 new offsprings
    """
    c1 = parent1['chromosome'].iloc[0].copy()
    c2 = parent2['chromosome'].iloc[0].copy()
    start = random.randint(0, len(c1)-1)
    end = random.randint(start, len(c1)-1)
    difference = list(set(c2) - set(c1[start:end+1]))
    order_q = []
    for i in range(len(difference)):
        heapq.heappush(order_q, (c2.index(difference[i]), difference[i]))
    # print(order_q)
    for i in range(len(c1)):
        if i < start or i > end:
            c1[i] = heapq.heappop(order_q)[1]
    # print(c1)
    return c1


def crossover2(parent1, parent2, fixed_pos=[]):
    """
    Perform recombination for order-based representation
    :return: dataframe with 2 new offsprings
    """
    len_fix = len(fixed_pos)
    c1 = parent1['chromosome'].iloc[0].copy()
    c2 = parent2['chromosome'].iloc[0].copy()
    a = random.randint(len_fix, len(c1))
    b = random.randint(len_fix, len(c1))
    while b == a:
        b = random.randint(len_fix, len(c1))
    new = c1[0:len_fix]
    c1_ex = c1[min(a,b):max(a,b)]
    new += c1_ex
    c2_ex = [item for item in c2 if item not in new]
    new += c2_ex
    return new

def crossover_one(parent1, parent2, city_table, fixed_pos=[]):
    c1 = crossover2(parent1, parent2, fixed_pos)
    fitness1 = cal_fitness(c1, city_table)
    cross_spring = pd.DataFrame({'chromosome': [c1], 'fitness': [fitness1]})
    return cross_spring

def crossover_all(parent1, parent2, city_table):
    """
    :param parent1: list
    :param parent2: list
    :param city_table: dataframe
    :return:
    """
    c1 = crossover(parent1, parent2)
    c2 = crossover(parent2, parent1)
    fitness1 = cal_fitness(c1, city_table)
    fitness2 = cal_fitness(c2, city_table)
    cross_spring = pd.DataFrame({'chromosome': [c1, c2], 'fitness': [fitness1, fitness2]})
    return cross_spring


def mutation(chromosome, fixed_pos=[]):
    """
    :param chromosome: List
    :return: List
    """
    len_fix = len(fixed_pos)
    point1 = random.randint(len_fix, len(chromosome) - 1)
    point2 = random.randint(len_fix, len(chromosome) - 1)
    while point1 == point2:
        point2 = random.randint(len_fix, len(chromosome) - 1)
    chromosome[point2], chromosome[point1] = chromosome[point1], chromosome[point2]
    return chromosome


def mutation_one(parent, city_table, fixed_pos=[]):
    parent1 = parent['chromosome'].iloc[0]
    f_offspring1 = mutation(parent1, fixed_pos)
    fitness1 = cal_fitness(f_offspring1, city_table)
    offspring = pd.DataFrame({'chromosome': [f_offspring1], 'fitness': [fitness1]})
    return offspring


def mutation_all(parent, city_table):
    """
    :param parent: dataframe
    :param city_table:
    :return:
    """
    parent1 = parent['chromosome'].iloc[0]
    parent2 = parent['chromosome'].iloc[1]
    f_offspring1 = mutation(parent1)
    f_offspring2 = mutation(parent2)
    fitness1 = cal_fitness(f_offspring1, city_table)
    fitness2 = cal_fitness(f_offspring2, city_table)
    offspring = pd.DataFrame({'chromosome': [f_offspring1, f_offspring2], 'fitness': [fitness1, fitness2]})
    return offspring


def new_offspring(parent1, parent2, para, city_table=city_10, fixed_pos=[]):
    """
    Version 3: This is the process of getting new_offspring with crossover, and mutation
    :param  Dataframe parent1: chromosome choosing randomly
    :param  Dataframe parent2: chromosome choosing randomly
    :return: Dataframe offspring
    """
    cross_prob = para['crossover_rate']
    mut_prob = para['mutation_rate']
    crossover_p = random.random()
    muatation_p = random.random()

    if crossover_p < cross_prob:
        offspring = crossover_one(parent1, parent2, city_table, fixed_pos)
    else:
        offspring = parent1

    if muatation_p < mut_prob:
        offspring = mutation_one(offspring, city_table, fixed_pos)

    return offspring


def next_generation(pop_table, para, city_table=city_10, fixed_pos=[]):
    """
    The process of one iteration of GA. VERSION 3
    :return:
    """
    pop_size = para['pop_size']
    next_gen = pd.DataFrame(columns=['chromosome', 'fitness'])
    best = pop_table.iloc[pop_table['fitness'].astype('float64').idxmax()]
    next_gen = next_gen.append(best)
    curr_c = 0
    # print(pop_size)
    while curr_c < pop_size-1:
        paren = selection(pop_table, par['k_tournament'])
        offsprin = new_offspring(paren[0:1], paren[1:2], par, city_table, fixed_pos)
        next_gen = next_gen.append(offsprin)
        curr_c += 1
    next_gen = next_gen.reset_index(drop=True)
    return next_gen


def GA_Process3(para, city_table=city_10, fixed_pos=[]):
    """
    :param para:
    :param city_table: the cities
    :param fixed_pos: the cities that must be travelled
    :return:
    """
    pop_table = init_pop(para['pop_size'], city_table, fixed_pos)
    # print('The initial table is ')
    # print(pop_table)
    cur_c = 0
    while cur_c < para['max_generation']:
        pop_table = next_generation(pop_table, para, city_table=city_10, fixed_pos=fixed_pos)
        print('Current iteration is ' + str(cur_c+1))
        optimal = pop_table.iloc[pop_table['fitness'].astype('float').idxmax()]
        print('the optimal distance is ' + str(100 / (optimal['fitness'])))
        print('The optimal solution is ' + str(optimal['chromosome']))
        print('')
        cur_c += 1
    return pop_table


def GA_process(para, city_table=city_10, fixed_pos=[]):
    """
    This is the function for version 2. NOT final one
    :return:
    """
    pop_table = init_pop(para['pop_size'], city_table, fixed_pos)
    # print('The initial population')
    # print(pop_table)
    epoch = 0
    avg_fit = []
    best_fit = []
    while epoch < para['max_generation']:
        print('The current epoch is ' + str(epoch))
        # print('The current len of population is ' + str(len(pop_table)))
        parent = selection(pop_table, para['k_tournament'])

        # Perform crossover if within probability
        crossover_p = random.random()
        if crossover_p < para['crossover_rate']:
            offspring = crossover_one(parent[0:1], parent[1:2], city_table,fixed_pos)
            # Perform mutation if within probability
            mutation_p = random.random()
            if mutation_p < para['mutation_rate']:
                offspring = mutation_one(offspring, city_table, fixed_pos)
            pop_table = pop_table.append(offspring, ignore_index=True)
        else:
            mutation_p = random.random()
            offspring = parent
            if mutation_p < para['mutation_rate']:
                offspring = mutation_one(offspring, city_table, fixed_pos)
                pop_table = pop_table.append(offspring, ignore_index=True)

        # print('The number of chromosome in population is :' + str(len(pop_table)))
        # Elitism
        if len(pop_table) == para['pop_size'] + 1:
            pop_table = pop_table.drop(pop_table['fitness'].astype('float64').idxmin())
            # pop_table = pop_table.drop(pop_table['fitness'].astype('float64').idxmin())
            pop_table = pop_table.reset_index(drop=True)
        epoch += 1
        mean_fit = pop_table['fitness'].mean()
        max_fit = pop_table['fitness'].max()
        avg_fit.append(mean_fit)
        best_fit.append(max_fit)
        print('The mean fitness of pop is '+ str(mean_fit))

        # Print out the optimal
        optimal = pop_table.iloc[pop_table['fitness'].astype('float').idxmax()]
        print('the optimal distance is ' + str(100/(optimal['fitness'])))
        # print(optimal)

    # print('The population table after genetic algorithm')
    # print(pop_table)
    optimal_solution = pop_table.iloc[pop_table['fitness'].astype('float').idxmax()]
    optimal_distance = str(100/(optimal['fitness']))
    return optimal_solution, optimal_distance, avg_fit, best_fit


def draw(para, optimal, avg_fit, best_fit):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(range(0, para['max_generation']), avg_fit, c="green")
    ax2.plot(range(0, para['max_generation']), best_fit, c="green")

    ax1.title.set_text('Average fitness score')
    ax2.title.set_text('Best fitness score')

    plt.show()


def add_more_city(org_table, num):
    for i in range(num):
        new_x = random.random()
        new_y = random.random()
        org_table.loc[-i-1] = [new_x, new_y]
    org_table = org_table.reset_index(drop=True)
    return org_table


def draw_route():
    return


def plot_cluster(data, label):
    """
    :param df: dataframe of the input data
    :param lab: label that outputted by model prediction
    :return:
    """
    data['label'] = label
    unique_l = np.unique(label)
    for i in unique_l:
        plt.scatter(data[data['label']==i]['x'], data[data['label']==i]['y'], label=i)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    random.seed(3)
    # Basic 10 city problem
    # Version 3 -----------------------------------
    # par = init_para(pop_size=100, max_gen=30, k_tournament=5, crossover_rate=0.85, mutation_rate=0.03)
    # GA_Process3(par, city_table=city_10, fixed_pos=[])
    # Version 2-----------------------------------
    # result = GA_process(par)
    # draw(par, result[0], result[2], result[3])

    # More city:
    # Version 3-----------------------------------
    # par = init_para(pop_size=100, max_gen=30, k_tournament=5, crossover_rate=0.85, mutation_rate=0.03)
    # more_city = add_more_city(city_10, 24)
    # GA_Process3(par, city_table=more_city, fixed_pos=[])
    # Version 2-----------------------------------
    # more_city = add_more_city(city_10, 24)
    # par = init_para(pop_size=300, max_gen=100, k_tournament=10, crossover_rate=0.95, mutation_rate=0.03)
    # result = GA_process(par, city_table=more_city)

    # Implied start and end city: Its a two-sequence ordering problem with adjustment of fitness value
    # Version 3-----------------------------------
    # par = init_para(pop_size=100, max_gen=30, k_tournament=5, crossover_rate=0.85, mutation_rate=0.03)
    # start_end = [8, 9]
    # GA_Process3(par, city_table=city_10, fixed_pos=start_end)
    # Version 2-----------------------------------
    # start_end = [6, 7]
    # result = GA_process(par, fixed_pos=start_end)
    # print(result[0])

    # EXTENSION 2: Sequential ordering problem
    # Version 3-----------------------------------
    # par = init_para(pop_size=100, max_gen=30, k_tournament=5, crossover_rate=0.85, mutation_rate=0.03)
    # fixed_point = [1,2,3]
    # GA_Process3(par, city_table=city_10, fixed_pos=fixed_point)
    # Version 2-----------------------------------
    # fixed_point = [6, 7, 8]
    # result = GA_process(par, fixed_pos=fixed_point)
    # print(result[0])



    # EXTENSION 4: Clustering problem
    data = pd.read_table('C:/Users/roder/Desktop/Project 1/Dataset/Cluster_dataset.txt', header=None,
                         delim_whitespace=True)
    data.columns = ["x", "y"]
    cluster_city = data.to_numpy()

    # kmeans = KMeans(n_clusters=3, max_iter=300)
    # label = kmeans.fit_predict(cluster_city)

    # model = Birch(threshold=0.1, n_clusters=3)
    # model.fit(cluster_city)
    # label = model.predict(cluster_city)

    # model = AffinityPropagation(damping=0.9)
    # model.fit(cluster_city)
    # label = model.predict(cluster_city)

    # model = DBSCAN(eps=0.03, min_samples=2)
    # label = model.fit_predict(cluster_city)

    model = SpectralClustering(n_clusters=3, random_state=10)
    label = model.fit_predict(cluster_city)

    plot_cluster(data, label)

