import json, time
import numpy as np
import pandas as pd

from datetime import datetime
from typing import List, Tuple
from tqdm import tqdm
from multiprocessing import Pool

from arg_parser import parse_args
from fuzzy import FuzzySystem
from chromosome import Chromosome
from test import comprehensive_test

def init_population(names: List[List[str]],
                    pop_size: int,
                    active_rules: float) -> List[Chromosome]:

    population = []
    for _ in range(pop_size):
        chromosome = Chromosome(names=names)
        chromosome.generate_random(active_rules=active_rules)
        population.append(chromosome)
    return population

def score_chromosome(fuzzy_system: FuzzySystem,
                     dataset: pd.DataFrame,
                     chromosome: Chromosome,
                     metric: str) -> float:

    ctrl_system = fuzzy_system.create_ctrl_system(chromosome)
    return fuzzy_system.compute_score(ctrl_system, dataset, metric)

def mutate(chromosome: Chromosome, mutation_rate: float, a_mutation: float) -> Chromosome:
    new_chromosome = Chromosome(names=chromosome.names)
    new_chromosome.x1 = chromosome.x1.copy()
    new_chromosome.x2 = chromosome.x2.copy()
    new_chromosome.y = chromosome.y.copy()
    new_chromosome.weights = chromosome.weights.copy()

    for i in range(new_chromosome.x1.size):
        # the whole rule should be change at once -> you cannot tweak rule continuously
        if np.random.rand() < mutation_rate:
            new_chromosome.x1[i] = np.random.randint(0, len(new_chromosome.x1_names[i]))
            new_chromosome.x2[i] = np.random.randint(0, len(new_chromosome.x2_names[i]))
            new_chromosome.y[i] = np.random.randint(0, len(new_chromosome.names[-1]))

        # weights are independent of rule values -> you can flip existing rule
        if np.random.rand() < a_mutation:
            new_chromosome.weights[i] = (new_chromosome.weights[i] + 1) % 2 # flip the rule

    return new_chromosome

def genetic_algorithm(fuzzy_system: FuzzySystem,
                      dataset: pd.DataFrame,
                      pop_size: int,
                      n_generations: int,
                      mutation: float,
                      active_rules: float,
                      a_mutation: float,
                      processes: int,
                      input: str,
                      metric: str) -> Tuple[Chromosome, float]:

    print(f'Training started ...')

    if input is None:
        print('Initial population was generated randomly')
        population = init_population(names=fuzzy_system.names, pop_size=pop_size, active_rules=active_rules)

        best_chromosome = None
        best_error = np.inf
    else:
        print('Initial population was loaded from file')
        chromosome = Chromosome(names=fuzzy_system.names)
        chromosome.load(input)
        population = [chromosome] * pop_size

        best_chromosome = population[0]
        best_error = score_chromosome(fuzzy_system, dataset, best_chromosome, metric)

    for i in range(n_generations):

        errors = np.empty(pop_size)
        if processes > 1:
            with Pool(processes=processes) as pool:
                futures = [
                    pool.apply_async(score_chromosome, (fuzzy_system, dataset, chrom, metric))
                    for chrom in population
                ]
                for idx, future in tqdm(enumerate(futures), total=pop_size, desc=f'Evaluating gen {i + 1}/{n_generations}'):
                    errors[idx] = future.get()
        else:
            for idx, chromosome in tqdm(enumerate(population), total=pop_size, desc=f'Evaluating Gen {i + 1}/{n_generations}'):
                errors[idx] = score_chromosome(fuzzy_system, dataset, chromosome, metric)

        current_best_chromosome = population[errors.argmin()]
        current_best_error = errors.min()

        changed = False
        if current_best_error <= best_error:
            best_error = current_best_error
            best_chromosome = current_best_chromosome
            changed = True

        # copy best parent to all children and mutate
        children = [best_chromosome] * pop_size
        children = [mutate(chromosome=child, mutation_rate=mutation, a_mutation=a_mutation) for child in children]
        population = children

        print(f'Best error so far: {best_error:.4f} [{metric}] ({"changed" \
            if changed else "not changed"}) - active rules: {best_chromosome.get_real_size()}')
        
        if not changed:
            print(f'Best error from current generation: {errors.min():.4f}')
        
        # chromosome if optimalization is cut short
        best_chromosome.save('tmp.npz')

    return best_chromosome, best_error

def main():
    args = parse_args()
    
    # set or create new seed
    seed = args.seed
    if args.seed is None:
        seed = int(time.time() * 1000000) % (2**32 - 1)
    np.random.seed(seed)
    print(f'Seed: {np.random.get_state()[1][0]}')
    
    # dataset must be in csv format and have exactly one 'target' column
    dataset = pd.read_csv(args.dataset)

    # generate names for membership functions (last is target)
    names = [['low', 'medium', 'high'] for _ in range(dataset.shape[1])]

    # -1 because last column is the target
    n_vars = len(names) - 1
    n_rules = n_vars * (n_vars - 1) // 2 # n choose k combinatory rule
    print(f'Number of rules: {n_rules} ({n_rules * 4} parameters)')

    # create fuzzy system - is same for all chromosomes
    fuzzy_system = FuzzySystem()
    fuzzy_system.initialize(dataset, names, 0.1) # step is hardcoded

    # training (GA)
    best_chromosome, best_error = genetic_algorithm(
        fuzzy_system=fuzzy_system,
        dataset=dataset,
        pop_size=args.pop_size,
        n_generations=args.generations,
        mutation=args.mutation,
        active_rules=args.active_rules,
        a_mutation=args.a_mutation,
        processes=args.processes,
        input=args.input,
        metric=args.error_metric
    )

    # save config for reproducibility
    config = vars(args)
    config['seed'] = int(np.random.get_state()[1][0])
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    with open(f'configs/config_{timestamp}.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # also save fuzzy system !!! hardcoded -> will rewrite file
    # this is because all runs on same dataset will have the same system
    fuzzy_system.save(f'system.json')

    # save best chromosome with same timestamp as config
    best_chromosome.save(f'chromosomes/chrom_{timestamp}.npz')

    # evaluation
    run_test(fuzzy_system, best_chromosome, dataset, args.error_metric, args.test)

def run_test(fuzzy_system: FuzzySystem,
             chromosome: Chromosome,
             dataset: pd.DataFrame,
             metric: str,
             test: bool) -> None:
    
    ctrl_system = fuzzy_system.create_ctrl_system(chromosome)

    print('<-----Evaluation results----->')
    if not test:
        error = fuzzy_system.compute_score(ctrl_system, dataset, metric)
        print(f'Error: {error:.4f}')
    else:
        comprehensive_test(fuzzy_system, chromosome, dataset)
    print('<---------------------------->')

if __name__ == '__main__':
    main()