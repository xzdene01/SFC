import numpy as np
import pandas as pd

from typing import List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from arg_parser import parse_args
from data_loader import load_data_split, load_data_split
from fuzzy import FuzzySystem
from chromosome import Chromosome

MULTI_THREAD = False

def init_population(names: List[List[str]], pop_size: int) -> List[Chromosome]:
    population = []
    for _ in range(pop_size):
        chromosome = Chromosome(names=names)
        chromosome.generate_random()
        population.append(chromosome)

    return population

def score_chromosome(dataset: pd.DataFrame,
                     names: List[List[str]],
                     chromosome: Chromosome) -> float:

    system = FuzzySystem(dataset=dataset, names=names)
    system.create_ctrl_system(chromosome)
    error, _, _ = system.compute_error(data=dataset, method='abs')
    return error

def select_parents(population: List[Chromosome], errors: np.ndarray, n_parents: int) -> List[Chromosome]:
    selected_indices = errors.argsort()[:n_parents]
    return [population[i] for i in selected_indices]

def crossover(parents: List[Chromosome], n_children: int) -> Chromosome:
    children = []
    for _ in range(n_children):
        p1, p2 = np.random.choice(parents, 2, replace=False)
        cross_point = np.random.randint(1, p1.x1.size)

        child = Chromosome(names=p1.names)
        child.x1 = np.concatenate([p1.x1[:cross_point], p2.x1[cross_point:]])
        child.x2 = np.concatenate([p1.x2[:cross_point], p2.x2[cross_point:]])
        child.y = np.concatenate([p1.y[:cross_point], p2.y[cross_point:]])

        children.append(child)

    return children

def mutate(chromosome: Chromosome, mutation_rate: float) -> Chromosome:
    n_mutations = int(mutation_rate * chromosome.x1.size)
    indices = np.random.choice(chromosome.x1.size, n_mutations, replace=False)

    # this will change the whole rule values
    for i in indices:
        chromosome.x1[i] = np.random.randint(0, len(chromosome.x1_names[i]))
        chromosome.x2[i] = np.random.randint(0, len(chromosome.x2_names[i]))
        chromosome.y[i] = np.random.randint(0, len(chromosome.names[-1]))
    
    return chromosome

def genetic_algorithm(dataset: pd.DataFrame,
                      names: List[List[str]],
                      pop_size: int,
                      n_generations: int,
                      mutation: float,
                      n_parents: int) -> Tuple[Chromosome, float]:

    population = init_population(names, pop_size)
    best_chromosome = None
    best_error = np.inf

    print(f'Training started ...')
    for i in range(n_generations):
        errors = np.empty(pop_size)
        if not MULTI_THREAD:
            for idx, chromosome in tqdm(enumerate(population), total=pop_size, desc=f"Evaluating Gen {i + 1}/{n_generations}"):
                errors[idx] = score_chromosome(dataset, names, chromosome)
        else:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(score_chromosome, dataset, names, chrom)
                    for chrom in population
                ]
                for idx, future in tqdm(enumerate(futures), total=pop_size, desc=f'Evaluating gen {i + 1}/{n_generations}'):
                    errors[idx] = future.result()

        changed = False
        if errors.min() < best_error:
            best_error = errors.min()
            best_chromosome = population[errors.argmin()]
            changed = True
        
        parents = select_parents(population=population, errors=errors, n_parents=n_parents)
        children = crossover(parents=parents, n_children=pop_size - n_parents)
        children = [mutate(chromosome=child, mutation_rate=mutation) for child in children]
        population = parents + children

        print(f'Generation {i + 1}/{n_generations}, best error: {best_error:.4f} ({"changed" \
            if changed else "not changed"}) - best from current generation: {errors.min():.4f}')
        
        best_chromosome.save('chromosomes/best_chromosome.npz')

    return best_chromosome, best_error

def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
    
    train, test = load_data_split(args.dataset, seed=args.seed)

    # generate names for membership functions (last must be the target)
    names = [['low', 'medium', 'high'] for _ in range(train.shape[1])]

    num_rules = len(names) * (len(names) - 1) // 2
    num_params = num_rules * 3
    print(f'Number of rules: {num_rules} ({num_params} parameters)')

    # training (GA)
    best_chromosome, best_error = genetic_algorithm(
        dataset=train,
        names=names,
        pop_size=args.pop_size,
        n_generations=args.generations,
        mutation=args.mutation,
        n_parents=args.parents
    )

    print(f'Best error (train): {best_error:.4f}')

    # evaluation
    run_test(dataset=test, names=names, chromosome=best_chromosome)

    print('Best chromosome rules:')
    best_chromosome.print(train.columns)

def run_test(dataset: pd.DataFrame, names: List[List[str]], chromosome: Chromosome):
    # create fuzzy system for chromosome
    system = FuzzySystem(dataset=dataset, names=names)
    system.create_ctrl_system(chromosome)

    # compute error for chromosome
    error, unfired, max_error = system.compute_error(dataset, 'abs')

    print('<-----Evaluation results----->')
    print(f'Error (test): {error:.4f}')
    print(f'Unfired rules: {unfired}')
    print(f'Max error: {max_error:.4f}')
    print('<---------------------------->')

if __name__ == '__main__':
    main()