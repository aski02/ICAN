# ICAN: Identifying confounders using Additive Noise Models

This is the repository for my bachelor thesis. It contains the entire codebase for the implementation, tests and experiments. It implements the algorithm from the paper "Identifying confounders using additive noise models" by Dominik Janzing, Jonas Peters, Joris Mooij and Bernhard Schölkopf (2012).

## Prerequisites

- **Python**: Version 3.10 or higher. Make sure that Python is compiled with support for Tkinter and SQLite, as these are required for `app.py` and `tests_hsic.py`, respectively. Typically, these are already included when installing Python.
- **pipenv**: Required for managing project dependencies.
- **R**: Only required for the Hilbert-Schmidt Independence Criterion unittest (`tests_hsic.py`). Make sure to install the "dHSIC" library via a R console session:
  ```
  install.packages("dHSIC")
  ```

## Installation

1. **Clone the repository**:
   ```
   git clone https://github.com/aski02/ICAN.git
   ```

2. **Navigate** to the project directory:
   ```
   cd ICAN
   ```

3. **Start the pipenv shell** and install the dependencies:
   ```
   pipenv shell
   pipenv install
   ```

## Running the Application

### Graphical User Interface

The algorithm can be run on a variety of different datasets with different parameter choices through this interface.

```
python3 app.py
```

### Unittests

The unittest for ican does not always correctly identify the causal structure. But this is to be expected, as we run it on few datapoints. The unittest for hsic should be successful in all cases.

```
python3 -m unittest tests_ican.py
python3 -m unittest tests_hsic.py
```

### Experiments

In order to recreate the experiments from the bachelor thesis, run the following commands. As the algorithm is very compute intensive, these experiments can take multiple hours to complete.

#### Old Faithful Geyser
In order to recreate the plots for the nonlinear causal discovery using ANMs on the Old Faithful Geyser dataset, run the following command. It saves the 4 created plots as `forward_fit.png`, `forward_residuals.png`, `backward_fit.png` and `backward_residuals.png`. The forward direction corresponds to: the duration of an eruption causes the length of the interval until the next eruption.
```
python3 cause_effect_visualization.py
```

#### Experiments for identifiability under increasing additive noise
The following command creates 10 plots and saves them like this: `noise_13_20.png`. The first number refers to the dataset: 13 is the dataset for X<-T->Y, 17 is the dataset for Y->X and 6 is the dataset for X->Y. The second number refers to the number of datapoints, i.e. 20, 40 or 60.
```
python3 experiment_noise.py
```

#### Experiments for evaluating parameter choices
The following command creates 6 plots and saves them like this: `dim_regression.png`. The two words refer to the two parameters which are compared, i.e. "dim" refers to "dimensionality reduction method" and "regression" refers to the "regression method". 
```
python3 experiment_variations.py
```

#### Performance analysis
The first command creates .txt files with the results and the second one visualizes them as follows. It creates 2 plots and saves them as `gpr_runtimes.png` and `xgb_runtimes.png`. The first one refers to the performance analysis for the default parameter choices, while the second one refers to the performance analysis for the choice of "XGBoost" and "Pearson correlation coefficient". Run the two commands in the defined order. 
```
python3 performance_analysis.py
python3 performance_visualization.py
```

## Project Overview

This project is structured as follows:

- `ican.py`: This script contains the implementation of the pseudocode from the paper.

- `datasets/`: This directory contains the raw data for the real-world datasets.

- `datasets.py`: All the datasets (both simulated and real-world) are accessible via this script.
  
- `app.py`: This script contains the graphical user interface.

- `cause_effect_visualization.py`: This script performs a method for causal discovery on nonlinear ANMs on the Old Faithful Geyser dataset and visualizes the results. The utilized method was described in the paper "Nonlinear causal discovery with additive noise models" by P. Hoyer, D. Janzing, J. M. Mooji, J. Peters and B. Schölkopf (2008).
  
- `eval.py`: This script evaluates a certain parameter combination for the algorithm and computes the accuracy as well as a custom score.

- `experiment_variations.py`: This script creates and visualizes the experiments for comparing different paramter choices. It utilizes `eval.py` to evaluate different combinations.

- `performance_analysis.py`: This script runs the performance analysis.

- `performance_visualization.py`: This script visualizes the results of the performance analysis.

- `hsic.py`: This script implements the Hilbert-Schmidt Independence Criterion. The code was sourced from a public GitHub repository: [HSIC](https://github.com/amber0309/HSIC)

- `tests_hsic.py`: This script contains unittests for comparing the HSIC implementation in `hsic.py` to the implementation of HSIC in R by CRAN.

- `tests_ican.py`: This script contains unittests for my implementation of the algorithm.
