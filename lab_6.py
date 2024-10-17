import random
import matplotlib.pyplot as plt
import numpy as np

POPULATION_SIZE = 50
MUTATION_RATE = 0.3
MAX_GENERATIONS = 1000


def display_chessboard(solution):
    chessboard = np.zeros((8, 8))
    chessboard[1::2, ::2] = 1  
    chessboard[::2, 1::2] = 1  
    fig, ax = plt.subplots()
    ax.imshow(chessboard, cmap='gray', interpolation='none')
    for col, row in enumerate(solution):
        ax.text(col, row, '♔', ha='center', va='center', fontsize=30, color='red')
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(np.arange(1, 9))
    ax.set_yticklabels(np.arange(1, 9))
    ax.grid(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.show()

def generate_individual():
    return [random.randint(0, 7) for _ in range(8)]

def generate_population(size):
    return [generate_individual() for _ in range(size)]

def fitness(individual):
    max_fitness = 28
    non_attacking = 0
    for i in range(8):
        for j in range(i + 1, 8):
            if individual[i] != individual[j] and abs(individual[i] - individual[j]) != abs(i - j):
                non_attacking += 1
                
    return non_attacking

def crossover(parent1, parent2):
    point_1 = random.randint(0, 7)
    point_2 = random.randint(0, 7)
    child1 = parent1[:min(point_1, point_2)] + parent2[min(point_1, point_2):max(point_1, point_2)] + parent1[max(point_1, point_2):]
    child2 = parent2[:min(point_1, point_2)] + parent1[min(point_1, point_2):max(point_1, point_2)] + parent2[max(point_1, point_2):]
    return child1, child2

def mutate(individual):
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, 7)
        individual[idx] = random.randint(0, 7)
    return individual

def genetic_algorithm():
    population = generate_population(POPULATION_SIZE)
    generation = 0

    while generation < MAX_GENERATIONS:
        population = sorted(population, key=lambda x: fitness(x), reverse=True)

        if fitness(population[0]) == 28:
            print(f"Solution found in generation {generation}")
            return population[0]
        
        new_population = []
        index = 0
        for _ in range(POPULATION_SIZE // 2):
            print(f"sample: {population[index]}")
            parent1, parent2 =  population[index], population[index + 1]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))
            # index += 2
        
        population = new_population
        generation += 1
    
    print("No solution found.")
    return None


solution = genetic_algorithm()
if solution:
    print("Solution:", solution)
    display_chessboard(solution)