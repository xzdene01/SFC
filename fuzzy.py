import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import List, Tuple

import matplotlib.pyplot as plt

from data_loader import load_data

class FuzzySystem:
    def __init__(self, dataset):
        self.dataset = dataset
        
        # definition (of structure of) input and output variables
        self.in_vars = []
        for column in self.dataset.columns:
            if column == 'class': # output muts be in 'class' column !!!
                continue
            self.in_vars.append(ctrl.Antecedent(np.arange(self.dataset[column].min(), \
                                                          self.dataset[column].max(), 0.1), column))
        
        # classes must be of integer type and must be in 'class' column
        # not here but for future purposes in definition of membership functions
        self.n_classes = len(self.dataset['class'].unique())
        self.out_var = ctrl.Consequent(np.arange(0, self.n_classes + 1.1, 0.1), 'class')
    
    def mem_funcs(self, names: List[List[str]] = None):
        # define default names for membership functions
        if names is None:
            names = [['low', 'medium', 'high'] for _ in range(len(self.in_vars))]
        self.names = names

        for i in range(len(self.in_vars)):
            self.in_vars[i].automf(names=self.names[i])
        
        # classes must be represented with numbers !!!
        for i in range(self.n_classes):
            self.out_var[i] = fuzz.trimf(self.out_var.universe, [i, i + 1, i + 2])
    
    def generate_rules(self, chromosome):
        i_chromosome = iter(chromosome)

        # apply chromosome to generate rules
        self.rules = []
        for i in range(len(self.in_vars)):
            for j in range(i + 1, len(self.in_vars)):
                var_a = self.in_vars[i]
                var_b = self.in_vars[j]

                val_a = self.names[i][i_chromosome.__next__()]
                val_b = self.names[j][i_chromosome.__next__()]
                val_out = i_chromosome.__next__()

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

def main():
    dataset = load_data()

    system = FuzzySystem(dataset)
    system.mem_funcs()

    # # show membership functions
    # system.in_vars[0].view()
    # system.out_var.view()
    # plt.show()

    # generate chromosome
    vars_len = len(system.in_vars)
    chromosome_len = (int)(vars_len * (vars_len - 1) / 2) * 3
    chromosome = []
    for i in range(chromosome_len):
        # every third element must class
        if i % 3 == 2:
            chromosome.append(np.random.randint(0, system.n_classes))
        else:
            chromosome.append(np.random.randint(0, len(system.names[0])))

    # define some example rules
    system.generate_rules(chromosome)
    print(f'Rules lenght: {len(system.rules)}')

    # get class from output
    class_centers = {
        'class1': 1,
        'class2': 2,
        'class3': 3
    }

    # compute fuzzy output
    simulation = system.compute(0)
    sys_output = simulation.output['class']
    print(f'Fuzzy output: {sys_output}')
    print(f'Prediction: {min(class_centers, key=lambda x: abs(class_centers[x] - sys_output))}')

    # show output membership function
    system.out_var.view(sim=simulation)
    plt.show()

if __name__ == '__main__':
    main()