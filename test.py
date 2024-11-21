import numpy as np
import pandas as pd

from scipy.stats import kstest, norm, expon, uniform, gamma, beta, anderson
from tqdm import tqdm

from chromosome import Chromosome
from fuzzy import FuzzySystem

# if ran from main
dataset_path = 'data/Concrete_Data.csv'
chromosome_path = 'chromosomes/best_chromosome.npz'

def print_results(target: pd.Series, predictions: list) -> None:
    # target stats
    mean = target.mean()
    variance = target.var()
    std_dev = target.std()

    # compute errors
    residuals = target - predictions
    mse = np.mean(residuals ** 2)
    mae = np.mean(np.abs(residuals))

    # compare metrics to variance
    mse_relative = mse / variance
    mae_relative = mae / std_dev

    # test normal distribution
    anderson_result = anderson(target)

    print('\nTarget variable stats:')
    print(f'\tMean: {mean:.4f}')
    print(f'\tVariance: {variance:.4f}')
    print(f'\tStandard deviation: {std_dev:.4f}')

    print('\nPrediction errors:')
    print(f'\tMSE: {mse:.4f}')
    print(f'\tMAE: {mae:.4f}')
    print(f'\tMSE relative to variance: {mse_relative:.4f}')
    print(f'\tMAE relative to standard deviation: {mae_relative:.4f}')

    print(f"\nAnderson-Darling Test: Statistic: {anderson_result.statistic:.2f}")
    for i, (sig, crit) in enumerate(zip(anderson_result.significance_level, anderson_result.critical_values)):
        print(f"\t{sig}% Significance Level:\tCritical Value={crit:.2f}")
    if anderson_result.statistic < anderson_result.critical_values[2]:  # 5% level
        print("The data appears to be normally distributed (5%).")
    else:
        print("The data does not appear to be normally distributed (5%).")

def get_predictions(row: pd.DataFrame,
                    system: FuzzySystem) -> float:
    misfires = 0
    predictions = []
    for i in tqdm(range(row.shape[0]), total=row.shape[0], desc='Getting predictions'):
        simulation = system.compute(row.iloc[i])
        try:
            predictions.append(simulation.output['target'])
        except:
            predictions.append(row['target'].mean())
            misfires += 1
    print(f'Misfired: {misfires}')
    return predictions

def comprehensive_test(dataset: pd.DataFrame, system: FuzzySystem) -> None:
    target = dataset['target']
    predictions = get_predictions(dataset, system)
    print_results(target, predictions)

def main():
    # load dataset
    dataset = pd.read_csv(dataset_path)

    # load chromosome
    names = [['low', 'medium', 'high'] for _ in range(dataset.shape[1])]
    chromosome = Chromosome(names=names)
    chromosome.load(chromosome_path)

    # print chromosome + rules
    print(chromosome)
    chromosome.print_rules(dataset.columns)

    # create fuzzy system
    system = FuzzySystem(dataset=dataset, names=names)
    system.create_ctrl_system(chromosome)

    comprehensive_test(dataset=dataset, system=system)

if __name__ == '__main__':
    main()