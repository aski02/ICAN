# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------

import timeit
import numpy as np
from ican import causal_inference
from datasets import generate_data

# -------------------------------------------------------------------------
# Performance Analysis
# -------------------------------------------------------------------------

def performance_analysis(dim_reduction="Isomap", neighbor_percentage=0.1, iterations=3, kernel="RBF", variance_threshold=3.0, independence_threshold=0.05, regression_method="GPR", independence_method="HSIC", min_distance="Nelder-Mead", min_projection="Nelder-Mead", dataset=0, filename=None):
    times = []

    dataset_sizes = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    for size in dataset_sizes:
        # Generate synthetic data
        T, X, Y = generate_data(size, dataset)
        X = X.reshape(-1,1)
        Y = Y.reshape(-1,1)
	
	# Start timer
        start_time = timeit.default_timer()

        result = causal_inference(X, Y,dim_reduction, neighbor_percentage, iterations, kernel, variance_threshold, independence_threshold, regression_method, independence_method, min_distance, min_projection)

        # Calculate the runtime
        runtime = timeit.default_timer() - start_time

        # Store the runtime
        times.append(runtime)
    
    with open(filename + ".txt", "w") as file:
        file.write("Runtimes:\n")
        for time in times:
            file.write(str(time) + "\n")

    return times

# -------------------------------------------------------------------------
# Run Performance Analysis
# -------------------------------------------------------------------------

performance_analysis(dim_reduction="Isomap", neighbor_percentage=0.1, iterations=3, kernel="RBF", variance_threshold=3.0, independence_threshold=0.05, regression_method="GPR", independence_method="HSIC", min_distance="Nelder-Mead", min_projection="Nelder-Mead", dataset=0, filename="runtimes_0_gpr")

performance_analysis(dim_reduction="Isomap", neighbor_percentage=0.1, iterations=3, kernel="RBF", variance_threshold=3.0, independence_threshold=0.05, regression_method="GPR", independence_method="HSIC", min_distance="Nelder-Mead", min_projection="Nelder-Mead", dataset=1, filename="runtimes_1_gpr")

performance_analysis(dim_reduction="Isomap", neighbor_percentage=0.1, iterations=3, kernel="RBF", variance_threshold=3.0, independence_threshold=0.05, regression_method="GPR", independence_method="HSIC", min_distance="Nelder-Mead", min_projection="Nelder-Mead", dataset=13, filename="runtimes_2_gpr")

performance_analysis(dim_reduction="Isomap", neighbor_percentage=0.1, iterations=3, kernel="RBF", variance_threshold=3.0, independence_threshold=0.05, regression_method="GPR", independence_method="HSIC", min_distance="Nelder-Mead", min_projection="Nelder-Mead", dataset=15, filename="runtimes_3_gpr")

performance_analysis(dim_reduction="Isomap", neighbor_percentage=0.1, iterations=3, kernel="RBF", variance_threshold=3.0, independence_threshold=0.05, regression_method="XGBoost", independence_method="Pearson", min_distance="Nelder-Mead", min_projection="Nelder-Mead", dataset=0, filename="runtimes_0_xgb")

performance_analysis(dim_reduction="Isomap", neighbor_percentage=0.1, iterations=3, kernel="RBF", variance_threshold=3.0, independence_threshold=0.05, regression_method="XGBoost", independence_method="Pearson", min_distance="Nelder-Mead", min_projection="Nelder-Mead", dataset=1, filename="runtimes_1_xgb")

performance_analysis(dim_reduction="Isomap", neighbor_percentage=0.1, iterations=3, kernel="RBF", variance_threshold=3.0, independence_threshold=0.05, regression_method="XGBoost", independence_method="Pearson", min_distance="Nelder-Mead", min_projection="Nelder-Mead", dataset=13, filename="runtimes_2_xgb")

performance_analysis(dim_reduction="Isomap", neighbor_percentage=0.1, iterations=3, kernel="RBF", variance_threshold=3.0, independence_threshold=0.05, regression_method="XGBoost", independence_method="Pearson", min_distance="Nelder-Mead", min_projection="Nelder-Mead", dataset=15, filename="runtimes_3_xgb")
