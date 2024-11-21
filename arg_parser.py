import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Fuzzy System')

    parser.add_argument('-d', '--dataset',      type=str, default='data.csv',   help='Path to dataset')
    parser.add_argument('-s', '--pop_size',     type=int, default=20,           help='Population size')
    parser.add_argument('-g', '--generations',  type=int, default=10,           help='Number of generations')
    parser.add_argument('-m', '--mutation',     type=float, default=0.01,       help='Mutation rate')
    parser.add_argument(      '--parents',      type=int, default=10,           help='Number of parents')
    parser.add_argument(      '--seed',         type=int, default=None,         help='Random seed')
    parser.add_argument('-p', '--processes',    type=int, default=1,            help='Number of processes')

    return parser.parse_args()