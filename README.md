# ICAN: Identifying confounders using Additive Noise Models

This is the repository for my bachelor thesis. It contains the entire codebase for the implementation, tests and experiments.

## Prerequisites

- **Python**: Version 3.10 or higher
- **pipenv**: required for managing project dependencies
- **R**: only required for the Hilbert-Schmidt Independence Criterion unittest (tests_hsic.py)

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

#### Experiments for identifiability under increasing additive noise
The following command creates 10 plots and saves them like this: "noise_13_20.png". The first number refers to the dataset: 13 is the dataset for X<-T->Y, 7 is the dataset for Y->X and 6 is the dataset for X->Y. The second number refers to the number of datapoints, i.e. 20, 40 or 60.
```
python3 experiment_noise.py

```

The following command creates 6 plots and saves them like this: "dim_regression.png". The two words refer to the two parameters which are compared, i.e. "dim" refers to "dimensionality reduction method" and "regression" refers to the "regression method". 
```
python3 experiment_variations.py
```

The first command creates .txt files with the results and the second one visualizes them as follows. It creates 2 plots and saves them as "gpr_runtimes.png" and "xgb_runtimes.png". The first one refers to the performance analysis for the default parameter choices, while the second one refers to the performance analysis for the choice of "XGBoost" and "Pearson correlation coefficient". Run the two commands in the defined order. 
```
python3 performance_analysis.py
python3 performance_visualization.py
```
