import json
import numpy as np
import skfuzzy as fuzz
import pandas as pd
import matplotlib.pyplot as plt

from skfuzzy import control as ctrl
from typing import List

from chromosome import Chromosome

class FuzzySystem:
    def __init__(self) -> None:
        self.names = None
        self.step = None
        self.default_output = None
        self.in_vars = None
        self.out_var = None

    def initialize(self, dataset: pd.DataFrame, names: List[List[str]], step: float) -> None:
        self.names = names
        self.step = step
        self.default_output = 0.0
        self.in_vars = []
        self.out_var = None
        
        self.default_output = dataset['target'].mean() # dataset must have 'target' column
        self.define_variables(dataset)
    
    def define_variables(self, dataset: pd.DataFrame) -> None:
        # input variables
        for column in dataset.columns:
            min_val = dataset[column].min()
            max_val = dataset[column].max()
            if column == 'target':
                self.out_var = ctrl.Consequent(np.arange(min_val, max_val + self.step, self.step), column)
                self.out_var.defuzzify_method = 'centroid'
            else:
                self.in_vars.append(ctrl.Antecedent(np.arange(min_val, max_val + self.step, self.step), column))

        # define membership functions
        self.define_membership_functions()
    
    def define_membership_functions(self) -> None:
        for i in range(len(self.in_vars)):
            self.in_vars[i].automf(names=self.names[i])
        self.out_var.automf(names=self.names[-1])

    def create_ctrl_system(self, chromosome: Chromosome) -> ctrl.ControlSystem:
        # create rule from chromosome for each pair of input variables
        iteraion = 0
        rules = []
        for i in range(len(self.in_vars)):
            for j in range(i + 1, len(self.in_vars)):
                chrom = chromosome[iteraion]

                # get variables to be set into rule (e.g. 'Cement', 'Water')
                var_a = self.in_vars[i]
                var_b = self.in_vars[j]

                # get values to be set into variables (e.g. 'low', 'medium', 'high')
                val_a = self.names[i][chrom['x1']]
                val_b = self.names[j][chrom['x2']]
                val_out = self.names[-1][chrom['y']]

                # create rule (e.g. 'Cement[low] AND Water[high] -> target[medium]')
                # if cement is low and water is high than target is medium
                rule = ctrl.Rule(var_a[val_a] & var_b[val_b], self.out_var[val_out])
                rule.weight = chrom['weight']
                rules.append(rule)
                iteraion += 1

        # create control system from rules
        control_system = ctrl.ControlSystem(rules)
        return control_system
    
    def compute(self, control_system: ctrl.ControlSystem, inputs: pd.Series) -> ctrl.ControlSystemSimulation:
        # create new simulation from control system
        simulation = ctrl.ControlSystemSimulation(control_system)

        # set input values
        for i in range(len(self.in_vars)):
            try:
                simulation.input[self.in_vars[i].label] = inputs.iloc[i]
            except: # variable is not in rules
                pass
        
        simulation.compute()
        return simulation

    def get_prediction(self, control_system: ctrl.ControlSystem, inputs: pd.Series) -> float:
        simulation = self.compute(control_system, inputs)
        try:
            return simulation.output['target']
        except:
            return self.default_output

    def compute_score(self, control_system: ctrl.ControlSystem, data: pd.DataFrame, metric: str) -> float:
        actual = data['target']
        predictions = [prediction for prediction in data.apply(lambda x: self.get_prediction(control_system, x), axis=1)]

        if metric == 'mae':
            return np.mean(np.abs(predictions - actual))
        elif metric == 'mse':
            return np.mean((predictions - actual) ** 2)

    ####################
    # helper functions
    ####################

    def save(self, filename):

        # this will not save membership functions nor universe -> these must be generated separately
        # universe could be saved but it is not necessary -> it can be generated from min, max and step
        def fuzzy_to_dict(fuzzy):
            return {
                'label': fuzzy.label,
                'min': float(fuzzy.universe.min()),
                'max': float(fuzzy.universe.max())
            }
        
        in_vars_serialized = [fuzzy_to_dict(var) for var in self.in_vars]
        out_var_serialized = fuzzy_to_dict(self.out_var)

        data = {
            'names': self.names,
            'float_step': self.step,
            'default_output': self.default_output,
            'in_vars': in_vars_serialized,
            'out_var': out_var_serialized
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    
    def load(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.names = data['names']
        self.step = data['float_step']
        self.default_output = data['default_output']

        def dict_to_fuzzy(data, is_output=False):
            universe = np.arange(data['min'], data['max'] + self.step, self.step)

            if is_output:
                fuzzy = ctrl.Consequent(universe, data['label'])
                fuzzy.defuzzify_method = 'centroid'
            else:
                fuzzy = ctrl.Antecedent(universe, data['label'])
            
            return fuzzy
        
        self.in_vars = [dict_to_fuzzy(var) for var in data['in_vars']]
        self.out_var = dict_to_fuzzy(data['out_var'], is_output=True)

        # mem funcs were not saved -> must be generated (could be cause theyre generated automatically)
        self.define_membership_functions()

    def print_vars(self, print_mem_funcs=False):
        print('Input variables:')
        for var in self.in_vars:
            print(f'{var.label}:')
            print(f'\tinterval: ({var.universe.min()}, {var.universe.max()})')
            print(f'\tstep: {var.universe[1] - var.universe[0]}')
            print(f'\tlenght: {var.universe.shape[0]}')

            if print_mem_funcs:
                var.view()
                plt.show()
        
        print('\nOutput variable:')
        print(f'{self.out_var.label}:')
        print(f'\tinterval: ({self.out_var.universe.min()}, {self.out_var.universe.max()})')
        print(f'\tstep: {self.out_var.universe[1] - self.out_var.universe[0]}')
        print(f'\tlenght: {self.out_var.universe.shape[0]}')

        if print_mem_funcs:
            self.out_var.view()
            plt.show()