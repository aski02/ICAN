# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from ican import causal_inference

# -------------------------------------------------------------------------
# Datasets
# -------------------------------------------------------------------------

# X->Y
def generate_data_6(n, noise):
    np.random.seed(seed=1)

    T = np.linspace(0, 1, n)
    Nx = np.random.uniform(-noise/6, 0.0, size=n)
    Ny = np.random.uniform(-noise, noise, size=len(T))

    X = T + Nx
    Y = -np.square(np.sin(T) - 0.5) + Ny

    return (T, X, Y)

# X<-T->Y
def generate_data_13(n, noise):
    np.random.seed(seed=5)

    T = np.linspace(0.1, 1, n)
    Nx = np.random.uniform(-noise, noise, size=len(T))
    Ny = np.random.uniform(-noise, noise, size=len(T))

    X = np.sin(2*T) * np.cos(2*T) + Nx
    Y = np.sin(T) * np.square(T - 0.3) + Ny

    return (T, X, Y)
    
# Y->X
def generate_data_17(n, noise):
    np.random.seed(seed=4)

    T = np.linspace(0.1, 1, n)
    Nx = np.random.normal(0, 7*noise, size=len(T))
    Ny = np.random.normal(0, noise, size=n)

    X = np.square(T) * np.log(2*T) * np.sin(T)+ Nx
    Y = 0.8 * T + Ny

    return (T, X, Y)

# -------------------------------------------------------------------------
# Visualizations
# -------------------------------------------------------------------------

def visualize(structures, noises, filename):
    unique_structures = ["No relationship", "No CAN model", "X->Y", "X<-T->Y", "Y->X"]
    data = []
    bin_size = 10
    for i in range(0, len(noises), bin_size):
        bin_structures = structures[i:i+bin_size]
        interval_str = f"[{np.round(noises[i], 3)}, {np.round(noises[i + bin_size - 1], 3)}]"
        for structure in unique_structures:
            data.append({"Noise": interval_str, "Structure": structure, "Count": bin_structures.count(structure)})

    df = pd.DataFrame(data)
    pivot_df = df.pivot(index="Structure", columns="Noise", values="Count")

    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------------------
# Experiments
# -------------------------------------------------------------------------

def experiment(generate_data, noises, true_structure, n, filename):
    structures = []
    results = []
    for noise in noises:
        _, X, Y = generate_data(n, noise)
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        
        T_hat, var, s1_hat, s2_hat, result, structure, p1, p2 = causal_inference(X=X, Y=Y, dim_reduction="Isomap", neighbor_percentage=0.1, iterations=3, kernel="RBF", variance_threshold=2.0, independence_threshold=0.05, regression_method="GPR", independence_method="HSIC", min_distance="Nelder-Mead", min_projection="Nelder-Mead")

        structures.append(structure)
        correct_count = 1 if structure == true_structure else 0
        results.append(correct_count)

    filename_txt = filename + ".txt"
    with open(filename_txt, "w") as file:
        file.write("Structures:\n")
        for structure in structures:
            file.write(str(structure) + "\n")
        file.write("Results:\n")
        for result in results:
            file.write(str(result) + "\n")
    
    visualize(structures, noises, filename)

# -------------------------------------------------------------------------
# Running experiments
# -------------------------------------------------------------------------

experiment(generate_data_6, np.linspace(0.001, 0.05, 100), "X->Y", 20, "noise_6_20")
experiment(generate_data_6, np.linspace(0.001, 0.05, 100), "X->Y", 40, "noise_6_40")
experiment(generate_data_6, np.linspace(0.001, 0.05, 100), "X->Y", 60, "noise_6_60")

experiment(generate_data_13, np.linspace(0.005, 0.05, 100), "X<-T->Y", 20, "noise_13_20")
experiment(generate_data_13, np.linspace(0.005, 0.05, 100), "X<-T->Y", 40, "noise_13_40")
experiment(generate_data_13, np.linspace(0.005, 0.05, 100), "X<-T->Y", 60, "noise_13_60")

experiment(generate_data_17, np.linspace(0.001, 0.01, 100), "Y->X", 20, "noise_17_20")
experiment(generate_data_17, np.linspace(0.001, 0.01, 100), "Y->X", 40, "noise_17_40")
experiment(generate_data_17, np.linspace(0.001, 0.01, 100), "Y->X", 60, "noise_17_60")
