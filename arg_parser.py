import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Fuzzy System')

    parser.add_argument('-d', '--dataset',      type=str, default='data.csv',   help='Path to dataset')
    parser.add_argument('-s', '--pop_size',     type=int, default=64,           help='Population size')
    parser.add_argument('-g', '--generations',  type=int, default=50,           help='Number of generations')
    parser.add_argument('-m', '--mutation',     type=float, default=0.1,        help='Mutation rate')
    parser.add_argument(      '--seed',         type=int, default=None,         help='Random seed')
    parser.add_argument('-p', '--processes',    type=int, default=1,            help='Number of processes')
    parser.add_argument('-t', '--test',         action='store_true',            help='Run comprehensive test')
    parser.add_argument('-i', '--input',        type=str, default=None,         help='Path to input chromosome (.npz)')

    # some extra arguments -> these will lead to bigger search space and slower convergence, but hopefully better results
    parser.add_argument(      '--active_rules', type=float, default=1.0,        help='Percentage of active rules')
    parser.add_argument(      '--a_mutation',   type=float, default=0.0,        help='Mutation rate for rule activation')

    return parser.parse_args()