import numpy as np
from typing import List

class Chromosome:
    def __init__(self, n_vars: int, n_classes: int, names: List[List[str]] = None):
        self.n_vars = n_vars
        self.n_classes = n_classes

        if names is None:
            names = [['low', 'medium', 'high'] for _ in range(n_vars)]
        self.names = names

        self.x1 = np.array([], dtype=int)
        self.x2 = np.array([], dtype=int)
        self.y = np.array([], dtype=int)

    def generate_random(self):
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                self.x1 = np.append(self.x1, np.random.randint(0, len(self.names[i])))
                self.x2 = np.append(self.x2, np.random.randint(0, len(self.names[j])))
                self.y = np.append(self.y, np.random.randint(1, self.n_classes + 1))
    
    def save(self, filename):
        np.savez(filename, x1=self.x1, x2=self.x2, y=self.y)
    
    def load(self, filename):
        loaded = np.load(filename)
        self.x1 = loaded['x1']
        self.x2 = loaded['x2']
        self.y = loaded['y']
    
    def print(self, var_list: List[str]):
        index = 0
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                x1_name = var_list[i]
                x2_name = var_list[j]

                x1_val = self.names[i][self.x1[index]]
                x2_val = self.names[j][self.x2[index]]
                y_val = self.y[index]

                print(f'Rule {index}: {x1_name}({x1_val}) & {x2_name}({x2_val}) -> class({y_val})')
                index += 1