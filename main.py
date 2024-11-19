import numpy as np
import pandas as pd

from typing import List, Tuple

from arg_parser import parse_args
from data_loader import load_data
from fuzzy import FuzzySystem
from chromosome import Chromosome

def score_chromosome(dataset: pd.DataFrame, chromosome: Chromosome) -> float:
    system = FuzzySystem(dataset)
    system.mem_funcs()

    error = system.error(chromosome)
    return error

def init_population(dataset: pd.DataFrame, pop_size: int) -> List[Chromosome]:
    population = []
    for _ in range(pop_size):
        chromosome = Chromosome(dataset.shape[1] - 1, len(dataset['class'].unique()))
        chromosome.generate_random()

        population.append(chromosome)

    return population

def select_parents(population: List[Chromosome], errors: np.ndarray, n_parents: int) -> List[Chromosome]:
    selected_indices = errors.argsort()[:n_parents]
    return [population[i] for i in selected_indices]

def crossover(parents: List[Chromosome], n_children: int) -> Chromosome:
    children = []
    for _ in range(n_children):
        p1, p2 = np.random.choice(parents, 2, replace=False)
        cross_point = np.random.randint(1, p1.x1.size)

        child = Chromosome(p1.n_vars, p1.n_classes)
        child.x1 = np.concatenate([p1.x1[:cross_point], p2.x1[cross_point:]])
        child.x2 = np.concatenate([p1.x2[:cross_point], p2.x2[cross_point:]])
        child.y = np.concatenate([p1.y[:cross_point], p2.y[cross_point:]])

        children.append(child)

    return children

def mutate(chromosome: Chromosome, mutation: float) -> Chromosome:
    index = 0
    for i in range(chromosome.n_vars):
        for j in range(i + 1, chromosome.n_vars):
            if np.random.rand() < mutation:
                chromosome.x1[index] = np.random.randint(0, len(chromosome.names[i]))
            if np.random.rand() < mutation:
                chromosome.x2[index] = np.random.randint(0, len(chromosome.names[j]))
            if np.random.rand() < mutation:
                chromosome.y[index] = np.random.randint(1, chromosome.n_classes + 1)
            index += 1

    return chromosome

def genetic_algorithm(dataset: pd.DataFrame,
                      pop_size: int,
                      n_generations: int,
                      mutation: float,
                      n_parents: int) -> Tuple[Chromosome, float]:
    population = init_population(dataset, pop_size)
    best_chromosome = None
    best_error = np.inf

    for i in range(n_generations):
        errors = np.array([score_chromosome(dataset, chrom) for chrom in population])
        if errors.min() < best_error:
            best_error = errors.min()
            best_chromosome = population[errors.argmin()]
        
        parents = select_parents(population, errors, n_parents)
        children = crossover(parents, pop_size - n_parents)
        children = [mutate(child, mutation) for child in children]
        population = parents + children

        print(f'Generation {i + 1}/{n_generations}, best error: {best_error:.4f}')

    return best_chromosome, best_error

def main():
    args = parse_args()
    dataset = load_data(args.dataset)

    # chromosome = Chromosome(dataset.shape[1] - 1, len(dataset['class'].unique()))
    # chromosome.generate_random()
    # error = score_chromosome(dataset, chromosome)
    # print(f'Error: {error:.4f}')

    best_chromosome, best_error = genetic_algorithm(
        dataset=dataset,
        pop_size=args.pop_size,
        n_generations=args.generations,
        mutation=args.mutation,
        n_parents=args.parents
    )

    print(f'Best chromosome: {best_chromosome}')
    print(f'Best error: {best_error:.4f}')

if __name__ == '__main__':
    main()