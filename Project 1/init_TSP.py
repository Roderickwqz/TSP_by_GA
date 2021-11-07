import numpy as np
import pandas as pd
import random
import math
import heapq
import matplotlib.pyplot as plt

numberOfCity = 10

city_10 = pd.DataFrame(np.array([[0.3642,0.7779], [0.7185,0.8312], [0.0986,0.5891], [0.2954,0.9606],
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


def generate_chromosome(numberOfCity):
    """
    generate a chromosome based on number of city. The chromosome has first and end value the same for indicating
    circular motion.
    :param numberOfCity: number of cities need to be traveled
    :return: List chromosome
    """
    chromosome = random.sample(range(0, numberOfCity), numberOfCity)
    # chromosome.append(chromosome[0])
    return chromosome


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
        dis = np.round(math.sqrt(((city_a_x - city_b_x)**2) + ((city_a_y - city_b_y)**2)), 5)
        fitness += dis
    # city_end_x = city_table.iloc[chromosome[-1], 0]
    # city_end_y = city_table.iloc[chromosome[-1], 1]
    # city_start_x = city_table.iloc[chromosome[0], 0]
    # city_start_y = city_table.iloc[chromosome[0], 0]
    # dis = np.round(math.sqrt(((city_end_x - city_start_x) ** 2) + ((city_end_y - city_start_y) ** 2)), 5)
    # fitness += dis
    # fitness = fitness/(len(chromosome))
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
    # print(c1)
    # print(c2)
    # print(c1[start:end+1])
    # print(difference)
    order_q = []
    for i in range(len(difference)):
        heapq.heappush(order_q, (c2.index(difference[i]), difference[i]))
    # print(order_q)
    for i in range(len(c1)):
        if i < start or i > end:
            c1[i] = heapq.heappop(order_q)[1]
    # print(c1)
    return c1


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


def mutation(chromosome):
    """
    :param chromosome: List
    :return: List
    """
    point1 = random.randint(0, len(chromosome) - 1)
    point2 = random.randint(0, len(chromosome) - 1)
    while point1 == point2:
        point2 = random.randint(0, len(chromosome) - 1)
    chromosome[point2], chromosome[point1] = chromosome[point1], chromosome[point2]
    return chromosome


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


def GA_process(para, city_table=city_10):
    pop_table = init_pop(para['pop_size'], city_table)
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
            offspring = crossover_all(parent[0:1], parent[1:2], city_table)
            # Perform mutation if within probability
            mutation_p = random.random()
            if mutation_p < para['mutation_rate']:
                offspring = mutation_all(offspring, city_table)
            pop_table = pop_table.append(offspring, ignore_index=True)
        else:
            mutation_p = random.random()
            offspring = parent
            if mutation_p < para['mutation_rate']:
                offspring = mutation_all(offspring, city_table)
                pop_table = pop_table.append(offspring, ignore_index=True)

        # print('The number of chromosome in population is :' + str(len(pop_table)))
        # Elitism
        if len(pop_table) == para['pop_size'] + 2:
            pop_table = pop_table.drop(pop_table['fitness'].astype('float64').idxmin())
            pop_table = pop_table.drop(pop_table['fitness'].astype('float64').idxmin())
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


def crossover2(parent1, parent2):
    """
    Perform recombination for order-based representation
    :return: dataframe with 2 new offsprings
    """
    c1 = parent1['chromosome'].iloc[0].copy()
    c2 = parent2['chromosome'].iloc[0].copy()
    start = random.randint(0, len(c1)-1)
    end = random.randint(start, len(c1)-1)
    c1_ex = c1[start:end]
    c2_ex = [item for item in c2 if item not in c1_ex]
    new = []
    p1 = 0
    p2 = 0
    for i in range(len(c1)):
        if i < start or i > end:
            new.append(c2_ex[p2])
            p2 += 1
        else:
            new.append(c1_ex[p1])
            p1 += 1
    return new


if __name__ == "__main__":
    random.seed(10)
    par = init_para(pop_size=10, max_gen=15, k_tournament=3, crossover_rate=0.85, mutation_rate=0.3)
    # result = GA_process(par)
    # draw(par, result[0], result[2], result[3])

    # more_city = add_more_city(city_10, 24)
    # result = GA_process(par)
    # print(result[0])
    pop_table = init_pop(par['pop_size'], city_10)
    print(pop_table)
    parent = selection(pop_table, par['k_tournament'])
    print(parent)


    offspring = crossover2(parent[0:1], parent[1:2])
    print(offspring)



