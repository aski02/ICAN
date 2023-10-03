# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------

from eval import compute_accuracy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Experiment
# -------------------------------------------------------------------------

def run_experiment(param1_name, param1_values, param2_name, param2_values, filename, ylabel, xlabel):
    accuracies = []
    scores = []
    results = []

    heatmap_data_accuracy = pd.DataFrame(index=param1_values, columns=param2_values)
    heatmap_data_score = pd.DataFrame(index=param1_values, columns=param2_values)

    for param1 in param1_values:
        for param2 in param2_values:
            kwargs = { 
                "dim_reduction": "Isomap", "neighbor_percentage": 0.1, "iterations": 3, "kernel": "RBF", 
                "variance_threshold": 2.0, "independence_threshold": 0.05, "regression_method": "GPR", 
                "independence_method": "HSIC", "min_distance": "Nelder-Mead", "min_projection": "Nelder-Mead"
            }
            kwargs[param1_name] = param1
            kwargs[param2_name] = param2
            
            accuracy, score, structures = compute_accuracy(**kwargs)
            
            results.append(structures)
            accuracies.append(accuracy)
            scores.append(score)
            
            heatmap_data_accuracy.at[param1, param2] = accuracy
            heatmap_data_score.at[param1, param2] = score
    
    with open(filename + ".txt", "w") as file:
        file.write("Accuracies:\n")
        for acc in accuracies:
            file.write(str(acc) + "\n")
        file.write("Scores:\n")
        for score in scores:
            file.write(str(score) + "\n")
        file.write("Results:\n")
        for res in results:
            file.write(str(res) + "\n")

    try:
        plt.figure(figsize=(10, 8))
        
        heatmap_data_accuracy = heatmap_data_accuracy.astype(float)
        heatmap_data_score = heatmap_data_score.astype(float)

        sns.heatmap(heatmap_data_accuracy, annot=True, cmap="YlGnBu")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(filename + "_accuracy_heatmap.png", bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data_score, annot=True, cmap="YlGnBu")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(filename + "_score_heatmap.png", bbox_inches="tight")
        plt.close()
    except ValueError as e:
        print("Error")

# -------------------------------------------------------------------------
# Run Experiments
# -------------------------------------------------------------------------

run_experiment("min_distance", ["Nelder-Mead", "L-BFGS-B", "BFGS", "powell"], "min_projection", ["Nelder-Mead" , "L-BFGS-B", "BFGS", "powell"], "min_min", "Minimization (Distance)", "Minimization (Dependence)")
run_experiment("independence_method", ["HSIC", "MI",  "Pearson", "Spearman", "Kendalltau"], "min_projection", ["Nelder-Mead" , "L-BFGS-B", "BFGS", "powell"], "dep_minproj", "Dependence measures", "Minimization (Dependence)")
run_experiment("dim_reduction", ["Isomap", "TSNE",  "LLE", "PCA", "KernelPCA"], "kernel", ["RBF", "RBF + White", "Matern", "Matern + White"], "dim_kernel", "Dimensionality reduction methods", "Kernel")
run_experiment("dim_reduction", ["Isomap", "TSNE",  "LLE", "PCA", "KernelPCA"], "regression_method", ["GPR", "DecisionTree", "RandomForest", "NuSVR", "XGBoost"], "dim_regression", "Dimensionality reduction methods", "Regression methods")
run_experiment("independence_method", ["HSIC", "MI",  "Pearson", "Spearman", "Kendalltau"], "kernel", ["RBF", "RBF + White", "Matern", "Matern + White"], "dep_kernel", "Dependence measures", "Kernel")
run_experiment("independence_method", ["HSIC", "MI",  "Pearson", "Spearman", "Kendalltau"], "regression_method", ["GPR", "DecisionTree", "RandomForest", "NuSVR", "XGBoost"], "dep_regression", "Dependence measures", "Regression methods")
