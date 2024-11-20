import numpy as np
from typing import List

class Chromosome:
    def __init__(self, names: List[List[str]]):
        self.len = 0
        self.names = names
        self.n_vars = len(self.names)

        # actual chromosome values
        self.x1 = np.array([], dtype=int)
        self.x2 = np.array([], dtype=int)
        self.y = np.array([], dtype=int)

        # possible names for each rule unpacked
        self.x1_names = []
        self.x2_names = []
        self.unpack_names(names)

    def unpack_names(self, names: List[List[str]]):
        for i in range(self.n_vars - 1):
            for j in range(i + 1, self.n_vars - 1):
                self.x1_names.append(names[i])
                self.x2_names.append(names[j])
                self.len += 1

    def generate_random(self):
        for i in range(self.len):
            self.x1 = np.append(self.x1, np.random.randint(0, len(self.x1_names[i])))
            self.x2 = np.append(self.x2, np.random.randint(0, len(self.x2_names[i])))
            self.y = np.append(self.y, np.random.randint(0, len(self.names[-1])))
    
    def save(self, filename):
        np.savez(filename, x1=self.x1, x2=self.x2, y=self.y)
    
    def load(self, filename):
        loaded = np.load(filename)
        self.x1 = loaded['x1']
        self.x2 = loaded['x2']
        self.y = loaded['y']
    
    def __str__(self):
        return f'x1: {self.x1}\nx2: {self.x2}\ny: {self.y}'
    
    # helper functions
    def print(self, var_list: List[str]):
        iter_x1 = iter(self.x1)
        iter_x2 = iter(self.x2)
        iter_y = iter(self.y)

        # we need to mimic rule creation to correctly map var names to values
        iteration = 0
        for i in range(self.n_vars - 1):
            for j in range(i + 1, self.n_vars - 1):
                x1_name = var_list[i]
                x2_name = var_list[j]

                x1_val = self.x1_names[iteration][next(iter_x1)]
                x2_val = self.x2_names[iteration][next(iter_x2)]
                y_val = self.names[-1][next(iter_y)]

                print(f'Rule {iteration}: {x1_name}({x1_val}) & {x2_name}({x2_val}) -> target({y_val})')
                iteration += 1