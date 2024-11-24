import argparse
import numpy as np
import pandas as pd

from scipy.stats import anderson
from tqdm import tqdm

from fuzzy import FuzzySystem
from chromosome import Chromosome

def print_results(target: pd.Series, predictions: list) -> None:
    # target stats
    mean = target.mean()
    variance = target.var()
    std_dev = target.std()

    # errors
    residuals = target - predictions
    mse = np.mean(residuals ** 2)
    mae = np.mean(np.abs(residuals))

    # compare metrics to variance
    mse_relative = mse / variance
    mae_relative = mae / std_dev
    r2 = 1 - (mse / variance)

    # test for normal distribution
    anderson_result = anderson(target)

    # print all results
    print('\nTarget variable stats:')
    print(f'\tMean: {mean:.4f}')
    print(f'\tVariance: {variance:.4f}')
    print(f'\tStandard deviation: {std_dev:.4f}')

    print('\nPrediction errors:')
    print(f'\tMSE: {mse:.4f}')
    print(f'\tMAE: {mae:.4f}')
    print(f'\tMSE relative to variance: {mse_relative:.4f}')
    print(f'\tMAE relative to standard deviation: {mae_relative:.4f}')
    print(f'\tR2: {r2:.4f}')

    print(f"\nAnderson-Darling Test: Statistic: {anderson_result.statistic:.2f}")
    for i, (sig, crit) in enumerate(zip(anderson_result.significance_level, anderson_result.critical_values)):
        print(f"\t{sig}% Significance Level:\tCritical Value={crit:.2f}")
    if anderson_result.statistic < anderson_result.critical_values[2]:  # 5% level
        print("The data appears to be normally distributed (5%).")
    else:
        print("The data does not appear to be normally distributed (5%)")

def comprehensive_test(fuzzy_system: FuzzySystem, chromosome: Chromosome, dataset: pd.DataFrame) -> None:
    ctrl_system = fuzzy_system.create_ctrl_system(chromosome)
    target = dataset['target']

    predictions = []
    for i in tqdm(range(len(dataset)), desc='Predicting'):
        prediction = fuzzy_system.get_prediction(ctrl_system, dataset.iloc[i])
        predictions.append(prediction)

    print_results(target, predictions)

def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset',      type=str, default='data/Concrete_Data.csv', help='Path to dataset')
    parser.add_argument('-f', '--fuzzy_system', type=str, default='system.json',            help='Path to fuzzy system')
    parser.add_argument('-c', '--chromosome',   type=str, default='chromosomes/chrom.npz',  help='Path to chromosome')
    args = parser.parse_args()

    dataset_path = args.dataset
    fuzzy_system_path = args.fuzzy_system
    chromosome_path = args.chromosome

    # load dataset
    dataset = pd.read_csv(dataset_path)

    # load fuzzy system
    fuzzy_system = FuzzySystem()
    fuzzy_system.load(fuzzy_system_path)

    # load chromosome
    chromosome = Chromosome(names=fuzzy_system.names)
    chromosome.load(chromosome_path)

    # print chromosome + rules
    print(chromosome)
    chromosome.print_rules(dataset.columns)

    comprehensive_test(fuzzy_system, chromosome, dataset)

if __name__ == '__main__':
    main()