import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Fuzzy System')

    parser.add_argument('--dataset',    type=str, default='data/Concrete_Data.csv', help='Path to dataset')
    parser.add_argument('--pop_size',   type=int, default=20,                       help='Population size')
    parser.add_argument('--generations',type=int, default=10,                       help='Number of generations')
    parser.add_argument('--mutation',   type=float, default=0.01,                   help='Mutation rate')
    parser.add_argument('--parents',    type=int, default=10,                       help='Number of parents')
    parser.add_argument('--seed',       type=int, default=None,                     help='Random seed')

    return parser.parse_args()