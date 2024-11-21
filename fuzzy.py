import numpy as np
import skfuzzy as fuzz
import pandas as pd
from skfuzzy import control as ctrl
from typing import List

import matplotlib.pyplot as plt

from chromosome import Chromosome

class FuzzySystem:
    def __init__(self,
                 dataset: pd.DataFrame,
                 names: List[List[str]],
                 float_step: float = 0.01,
                 full: bool = True) -> None:
        
        self.names = names
        self.in_vars = []
        self.out_var = None
        self.control_system = None
        
        if full:
            self.define_variables(dataset=dataset, float_step=float_step)
            self.mem_funcs()
    
    def define_variables(self, dataset: pd.DataFrame, float_step: float = 0.01):
        # dataset must have 'target' column that will be used as output variable
        if 'target' not in dataset.columns:
            print('E: dataset does not have target column')
            exit(1)

        # input variables
        for column in dataset.columns:
            if column == 'target':
                continue

            dtype = dataset[column].dtype
            if dtype != 'float64' and dtype != 'int64':
                print(f'E: {column} is not of type float or int')
                exit(1)

            min_val = dataset[column].min()
            max_val = dataset[column].max()
            step = float_step if dtype == 'float64' else 1
            self.in_vars.append(ctrl.Antecedent(np.arange(min_val, max_val + step, step), column))
        
        # output variable
        min_val = dataset['target'].min()
        max_val = dataset['target'].max()
        step = float_step if dataset['target'].dtype == 'float64' else 1
        self.out_var = ctrl.Consequent(np.arange(min_val, max_val + step, step), 'target')

        # defuzzification method (centroid is default)
        self.out_var.defuzzify_method = 'centroid'

    def mem_funcs(self):
        for i in range(len(self.in_vars)):
            self.in_vars[i].automf(names=self.names[i])
        self.out_var.automf(names=self.names[-1])
    
    def create_ctrl_system(self, chromosome: Chromosome):
        # create rule from chromosome for each pair of input variables
        iteraion = 0
        rules = []
        for i in range(len(self.in_vars)):
            for j in range(i + 1, len(self.in_vars)):
                chrom = chromosome[iteraion]

                # get variables to be set into rule (eg. 'Cement', 'Water')
                var_a = self.in_vars[i]
                var_b = self.in_vars[j]

                # get values to be set into variables (eg. 'low', 'medium', 'high')
                val_a = self.names[i][chrom['x1']]
                val_b = self.names[j][chrom['x2']]
                val_out = self.names[-1][chrom['y']]

                # create rule (eg. 'Cement[low] AND Water[high] -> target[medium]')
                # if cement is low and water is high than target is medium
                rule = ctrl.Rule(var_a[val_a] & var_b[val_b], self.out_var[val_out])
                rule.weight = chrom['weight']
                rules.append(rule)
                iteraion += 1

        # create control system from rules
        self.control_system = ctrl.ControlSystem(rules)
        return self.control_system
    
    def compute(self, inputs: pd.Series) -> ctrl.ControlSystemSimulation:
        # check for control system
        if self.control_system is None:
            print('E: control system not defined - cannot compute')
            exit(1)

        # create new simulation from control system
        simulation = ctrl.ControlSystemSimulation(self.control_system)

        # set input values
        for i in range(len(self.in_vars)):
            try:
                simulation.input[self.in_vars[i].label] = inputs.iloc[i]
            except: # variable is not in rules
                pass
        
        # make computation
        simulation.compute()
        return simulation

    def compute_error(self, data: pd.DataFrame, method: str = 'abs'):
        # check for control system
        if self.control_system is None:
            print('E: control system not defined - cannot compute error')
            exit(1)
        
        # compute output for each row and accumulate error
        acc_error = unfired = max_error = 0
        for i in range(len(data)):
            simulation = self.compute(data.iloc[i])

            try:
                output = simulation.output['target']
            except:
                # no rules were fire -> none exist for these inputs
                if simulation.output == {}:
                    unfired += 1
                else:
                    print('E: output not found')
                    print(simulation.output)
                    exit(1)
                continue
            
            error = 0
            if method == 'abs':
                error = abs(output - data['target'][i])
            elif method == 'sqrt':
                error = (output - data['target'][i]) ** 2
            else:
                print(f'E: method {method} not implemented')
                exit(1)
            
            if error > max_error:
                max_error = error

            acc_error += error
        
        # add worst case scenario for unfired rules
        acc_error += unfired * max_error
        return acc_error / len(data), unfired, max_error

    # helper functions
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