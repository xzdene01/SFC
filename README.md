# SFC - Soft Computing Project

## About

- author: Jan Zdeněk (xzdene01)
- mail: <xzdene01@vutbr.cz>
- school: BUT FIT (VUT FIT)
- subject: SFC - Soft Computing
- project variant: 11. GA+Fuzzy

Main focus of this project is to find best rules for fuzzy inference using genetic algorithm. These rules will be used to predict value of one output variable from input variables.

## Instalation

### Automatic with script

For automatic instalation just run command:

```bash
source install.sh
```

This script will:

- *install miniconda* locally inside project root directory
- *create environment* from environment.yaml file
- *activate environment* that was just created

### Manual with conda environment

If you already have conda installed just run this command:

```bash
conda env create --file environment.yaml --name sfc
```

## Usage

### Optimalization

For optimalization with default parameters you can simply run:

```bash
python main.py
```

If you need to specify some parameters you can run the previously mentioned command with ``-h`` switch, this will print help on stdout.

```text
main.py [-h] [-d DATASET] [-s POP_SIZE] [-g GENERATIONS] [-m MUTATION] [--seed SEED] [-p PROCESSES] [-t] [-i INPUT] [-e {mse,mae}] [--active_rules ACTIVE_RULES] [--a_mutation A_MUTATION]

options:

    -h, --help                          Show this help message and exit
    -d, --dataset       DATASET         Path to dataset
    -s, --pop_size      POP_SIZE        Population size
    -g, --generations   GENERATIONS     Number of generations
    -m, --mutation      MUTATION        Mutation rate
        --seed          SEED            Random seed
    -p, --processes     PROCESSES       Number of processes
    -t, --test                          Run comprehensive test
    -i, --input         INPUT           Path to input chromosome (.npz)
    -e, --error_metric  {mse,mae}       Error metric to use when computing fitness
        --active_rules  ACTIVE_RULES    Percentage of active rules
        --a_mutation    A_MUTATION      Mutation rate for rule activation
```

### Evaluation

For evaluation with default parameters you can simply run:

```bash
python test.py
```

If you need to specify some parameters you can run the previously mentioned command with ``-h`` switch, this will print help on stdout.

```text
usage: test.py [-h] [-d DATASET] [-f FUZZY_SYSTEM] [-c CHROMOSOME]

options:
    -h, --help                          Show this help message and exit
    -d, --dataset       DATASET         Path to dataset
    -f, --fuzzy_system  FUZZY_SYSTEM    Path to fuzzy system
    -c, --chromosome    CHROMOSOME      Path to chromosome
```
