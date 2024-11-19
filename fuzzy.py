import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import List

import matplotlib.pyplot as plt

from chromosome import Chromosome

class FuzzySystem:
    def __init__(self, dataset):
        self.dataset = dataset
        
        # definition (of structure of) input and output variables
        self.in_vars = []
        for column in self.dataset.columns:
            if column == 'class': # output muts be in 'class' column !!!
                continue
            self.in_vars.append(ctrl.Antecedent(np.arange(self.dataset[column].min(),
                                                self.dataset[column].max(), 0.1), column))
        
        # classes must be of integer type and must be in 'class' column
        # not here but for future purposes in definition of membership functions
        self.n_classes = len(self.dataset['class'].unique())
        self.out_var = ctrl.Consequent(np.arange(1, self.n_classes + 0.1, 0.1), 'class')
    
    def mem_funcs(self, names: List[List[str]] = None):
        # define default names for membership functions
        if names is None:
            names = [['low', 'medium', 'high'] for _ in range(len(self.in_vars))]
        self.names = names

        for i in range(len(self.in_vars)):
            self.in_vars[i].automf(names=self.names[i])

        # classes must be represented with numbers !!!
        self.out_var.automf(names=[str(i) for i in range(1, self.n_classes + 1)])
    
    def generate_rules(self, chromosome: Chromosome):
        # apply chromosome to generate rules
        index = 0
        self.rules = []
        for i in range(len(self.in_vars)):
            for j in range(i + 1, len(self.in_vars)):
                var_a = self.in_vars[i]
                var_b = self.in_vars[j]

                val_a = self.names[i][chromosome.x1[index]]
                val_b = self.names[j][chromosome.x2[index]]
                val_out = str(chromosome.y[index])

                rule = ctrl.Rule(var_a[val_a] & var_b[val_b], self.out_var[val_out])
                self.rules.append(rule)
        
        # create control system from rules
        self.control_system = ctrl.ControlSystem(self.rules)
    
    def compute(self, row: int):
        simulation = ctrl.ControlSystemSimulation(self.control_system)

        for column in self.dataset.columns:
            if column == 'class':
                continue
            simulation.input[column] = self.dataset[column][row]
        
        simulation.compute()

        return simulation

    def error(self, chromosome: Chromosome):
        self.generate_rules(chromosome)

        fitness = 0
        not_found = 0
        for i in range(len(self.dataset)):
            simulation = self.compute(i)
            try:
                output = simulation.output['class']
            except:
                not_found += 1
            fitness += abs(output - self.dataset['class'][i])
        return fitness / (len(self.dataset) - not_found)