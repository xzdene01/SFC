import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import List, Tuple

import matplotlib.pyplot as plt

from data_loader import load_data

def generate_rules(in_vars, output, in_bins: Tuple, out_bins: Tuple) -> List[ctrl.Rule]:
    rules = []
    for var_a in in_vars:
        for var_b in in_vars:
            if var_a == var_b:
                continue

            # random choice from bins
            rnd_a = np.random.choice(in_bins)
            rnd_b = np.random.choice(in_bins)
            rnd_out = np.random.choice(out_bins)

            rule = ctrl.Rule(var_a[rnd_a] & var_b[rnd_b], output[rnd_out])
            rules.append(rule)
    
    return rules

def main():
    dataset = load_data()
    
    # define the input and output fuzzy variables
    in_vars = []
    for column in dataset.columns:
        if column == 'class':
            continue
        in_vars.append(ctrl.Antecedent(np.arange(dataset[column].min(), dataset[column].max(), 0.1), column))
    
    output = ctrl.Consequent(np.arange(0, 4.1, 0.1), 'class')

    # define the membership functions
    names = ['low', 'medium', 'high']
    for in_var in in_vars:
        in_var.automf(names=names)
    
    output['class1'] = fuzz.trimf(output.universe, [0, 1, 2])
    output['class2'] = fuzz.trimf(output.universe, [1, 2, 3])
    output['class3'] = fuzz.trimf(output.universe, [2, 3, 4])

    # generate list of random choices that is len(in_vars) ^ 2 * 3 long
    choices = np.random.randint(0, 3, len(in_vars) ** 2 * 3)

    # define some example rules
    rules = generate_rules(in_vars, output, ('low', 'medium', 'high'), ('class1', 'class2', 'class3'))

    # generate rnd list of weights for rules
    weights = np.random.rand(len(rules))

    # apply weights to rules
    for i, rule in enumerate(rules):
        if i > len(weights) / 2:
            rule.weight = 0
        else:
            rule.weight = weights[i]

    # create the control system
    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)

    # provide inputs from first row of dataset
    for column in dataset.columns:
        if column == 'class':
            continue
        sim.input[column] = dataset[column][0]

    # compute fuzzy output
    sim.compute()

    # get class from output
    class_centers = {
        'class1': 1,
        'class2': 2,
        'class3': 3
    }

    sim_output = sim.output['class']
    prediction = min(class_centers, key=lambda x: abs(class_centers[x] - sim_output))

    print(f'Fuzzy output: {sim_output}')
    print(f'Prediction: {prediction}')

if __name__ == '__main__':
    main()