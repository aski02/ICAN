# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from datasets import generate_data
from hsic import hsic_gam
from matplotlib import pyplot as plt
import numpy as np

# -------------------------------------------------------------------------
# Visualizations
# -------------------------------------------------------------------------

def plot_residuals(X, residuals, xlabel, ylabel, filename):
    fig, ax = plt.subplots()
    
    ax.scatter(X, residuals, s=10, color='blue', alpha=0.5, label='Residuals')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    plt.savefig(filename)
    plt.close(fig)

def plot_curve(X, Y, model, xlabel, ylabel, filename):
    fig, ax = plt.subplots()
    
    ax.scatter(X, Y, s=10, color='blue', alpha=0.5, label='Data points')
    sorted_indices = np.argsort(X, axis=0)
    X_sorted = X[sorted_indices].reshape(-1, 1)
    predicted_Y = model.predict(X_sorted).reshape(-1, 1)
    ax.plot(X_sorted, predicted_Y, "r--", linewidth=1.2, label='Fitted curve')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    plt.savefig(filename)
    plt.close(fig)


# -------------------------------------------------------------------------
# Algorithm
# -------------------------------------------------------------------------

def check_model(X, Y, threshold, xlabel, ylabel, filename_fit, filename_residuals):
    kernel = RBF(length_scale_bounds="fixed", length_scale=10.0)
    model = GaussianProcessRegressor(kernel=kernel)
    model.fit(X, Y)
    
    X.sort()
    Y.sort()
    residuals = Y.reshape(-1,1) - model.predict(X).reshape(-1,1)
    plot_curve(X, Y, model, xlabel, ylabel, filename_fit)
    plot_residuals(X, residuals, xlabel, 'residuals', filename_residuals)
    
    testStat, thresh, p = hsic_gam(X, residuals, alph=threshold)
    
    if testStat < thresh:
        return True, np.round(p, 4)
    else:
        return False, np.round(p, 4)

T, X, Y = generate_data(250, 3)
X, Y = X.reshape(-1,1), Y.reshape(-1,1)

result1, p1 = check_model(X=X, Y=Y, threshold=0.05, xlabel="duration", ylabel="interval", filename_fit='forward_fit.png', filename_residuals='forward_residuals.png')
result2, p2 = check_model(X=Y, Y=X, threshold=0.05, xlabel="interval", ylabel="duration", filename_fit='backward_fit.png', filename_residuals='backward_residuals.png')
print(p1, p2)
