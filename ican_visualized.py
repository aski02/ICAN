import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.manifold import Isomap
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from matplotlib.backends.backend_pdf import PdfPages
from datasets import generate_data
from sklearn.gaussian_process.kernels import ExpSineSquared
from hsic import hsic_gam
from sklearn.neighbors import NearestNeighbors

# Create pdf
pdf_pages = PdfPages('visualized.pdf')

# Use NN to draw curve because some of our data is not injective
def plotCurves(X, Y, style):
    data = np.column_stack([X, Y])	# Needed for NN
    
    # Use NearestNeighbors to find two closest points
    nbrs = NearestNeighbors(n_neighbors=3, algorithm="ball_tree").fit(data)
    distances, indices = nbrs.kneighbors(data)

    # Draw lines to two closest neighbors for each point (assuming they are the left and right neighbors)
    for i in range(len(data)):
    	for j in range(1, 3):
    		plt.plot([data[i,0], data[indices[i,j],0]], [data[i,1], data[indices[i,j],1]], style)
    		
def plotProjections(X, Y, T, T_hat):
    plt.figure(figsize=(8, 6))
    plt.title("True/Estimated curves with projections")
    plt.scatter(X, Y, s=10)
    
    T_hat.sort()
    predicted_X = s1_hat.predict(np.linspace(T_hat[0], T_hat[-1], 2000)).reshape(-1,1)
    predicted_Y = s2_hat.predict(np.linspace(T_hat[0], T_hat[-1], 2000)).reshape(-1,1)
    
    T_copy = np.linspace(T[0], T[-1], 2000).reshape(-1,1)
    plotCurves(predicted_X, predicted_Y, "r-")
    plotCurves(np.log(T_copy) * T_copy, np.square(T_copy), "k-")
    
    for i in range(5, len(T), len(T) // 7):
    	noisy_x = X[i]
    	noisy_y = Y[i]
    	
    	# Projection to true data point
    	true_x = np.log(T[i]) * T[i]
    	true_y = np.square(T[i])
    	plt.plot([true_x, noisy_x], [true_y, noisy_y], "k-")
    	
    	# Projection minimizing l2 distance
    	distances = np.sqrt(np.square((predicted_X - noisy_x)) + np.square((predicted_Y - noisy_y)))
    	min_idx = np.argmin(distances)
    	min_x, min_y = predicted_X[min_idx], predicted_Y[min_idx]
    	plt.plot([min_x, noisy_x], [min_y, noisy_y], "r-")
    
    plt.xlabel("X")
    plt.ylabel("Y")
    pdf_pages.savefig()

def plotData(X, Y, title, labelX="X", labelY="Y"):
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='blue')
    plt.title(title)
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    pdf_pages.savefig()

def plotConfounder(X, Y, T, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(T, X, color="blue", label="X")
    plt.scatter(T, Y, color="green", label="Y")
    plt.title(title)
    plt.xlabel("T")
    plt.legend()
    pdf_pages.savefig()

def plotResiduals(Nx, Ny, T):
    plt.figure(figsize=(8, 6))
    plt.scatter(T, Nx, color="blue")
    plt.xlabel("estimated T")
    plt.ylabel("estimated Nx")
    pdf_pages.savefig()

    plt.figure(figsize=(8, 6))
    plt.scatter(T, Ny, color="blue")
    plt.xlabel("estimated T")
    plt.ylabel("estimated Ny")
    pdf_pages.savefig()

    plt.figure(figsize=(8, 6))
    plt.scatter(Nx, Ny, color="blue")
    plt.xlabel("estimated Nx")
    plt.ylabel("estimated Ny")
    pdf_pages.savefig()

# paper used Isomap
def dimReduction(X, Y):
    X = X.flatten()
    Y = Y.flatten()
    data = np.concatenate((X[:, np.newaxis], Y[:, np.newaxis]), axis=1)  # Isomap requires 2 columns (samples x dimensions)

    neighbors = int(X.shape[0] * 0.1)  # high value: more neighbors are taken into account => smoother curve (less complexity)

    iso = Isomap(n_components=1, n_neighbors=neighbors)  # n_components = 1 because T_hat is 1-dimensional
    iso.fit(data)
    T_hat = iso.transform(data)

    plotConfounder(X, Y, T_hat, "Confounder after Isomap")

    return T_hat

# using the method proposed in the paper
def fitCurve(X, Y):
    # Initial dimensionality reduction
    T_hat = dimReduction(X, Y)

    kernel = 1.0 * ExpSineSquared(1.0, 5.0) + WhiteKernel(1e-1)
    s1_hat = GaussianProcessRegressor(kernel=kernel)
    s2_hat = GaussianProcessRegressor(kernel=kernel)

    for _ in range(5):
        # Step 1: Estimate s_hat using Gaussian Process Regression
        s1_hat.fit(T_hat, X)
        s2_hat.fit(T_hat, Y)

        # Step 2: Update T_hat so that l2-distance is minimized
        def l2dist(T):
            T = T.reshape(-1, 1)
            return np.linalg.norm([s1_hat.predict(T).reshape(-1, 1) - X, s2_hat.predict(T).reshape(-1, 1) - Y])

        # Minimize l2-distance w.r.t. T_hat
        init_guess = T_hat.flatten()
        T_hat_new = minimize(l2dist, init_guess, method="L-BFGS-B").x.reshape(-1, 1)

        # Check for convergence
        if np.linalg.norm(T_hat_new - T_hat) / np.linalg.norm(T_hat) < 1e-4:
            break

        T_hat = T_hat_new

    return s1_hat, s2_hat, T_hat

# paper used Hilbert-Schmidt Independence Criterion
def dep(X1, X2):
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(-1, 1)

    testStat, thresh, p = hsic_gam(X1, X2, alph=0.05)
    
    return testStat < thresh  # is true if X1, X2 are independent

def areIndependent(T_hat, Nx_hat, Ny_hat):
    return dep(Nx_hat, Ny_hat) and dep(Nx_hat, T_hat) and dep(Ny_hat, T_hat)

def projection(T_hat, s1_hat, s2_hat, X, Y):
    def depSum(T):
        T = T.reshape(-1, 1)

        Nx_hat = X - s1_hat.predict(T).reshape(-1, 1)
        Ny_hat = Y - s2_hat.predict(T).reshape(-1, 1)

        testStat1, thresh1, p1 = hsic_gam(Nx_hat, Ny_hat, alph=0.05)
        testStat2, thresh2, p2 = hsic_gam(Nx_hat, T, alph=0.05)
        testStat3, thresh3, p3 = hsic_gam(Ny_hat, T, alph=0.05)

        score1 = testStat1 - thresh1
        score2 = testStat2 - thresh2
        score3 = testStat3 - thresh3

        return score1 + score2 + score3

    # Minimize dependence w.r.t. T_hat
    init_guess = T_hat.flatten()
    T_hat = minimize(depSum, init_guess, method="L-BFGS-B").x.reshape(-1, 1)

    return T_hat

# paper used non-linear regression (no particular method specified)
def regressionGPR(T_hat, X, Y, Nx_hat, Ny_hat):
    kernel = 1.0 * ExpSineSquared(1.0, 5.0) + WhiteKernel(1e-1)
    
    s1_hat = GaussianProcessRegressor(kernel=kernel)
    s2_hat = GaussianProcessRegressor(kernel=kernel)

    s1_hat.fit(T_hat, X - Nx_hat)
    s2_hat.fit(T_hat, Y - Ny_hat)

    return s1_hat, s2_hat

# currently not used
def regressionPoly(T_hat, X, Y, Nx_hat, Ny_hat, deg=3):
    s1_hat = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    s1_hat.fit(X - Nx_hat, T_hat)

    s2_hat = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    s2_hat.fit(Y - Ny_hat, T_hat)

    return s1_hat, s2_hat

# ICAN algorithm
def identify_confounders(X, Y, K=5):  # paper used K = 5000 (but if successful then termination usually occurs within 1-2 iterations)
    s1_hat, s2_hat, T_hat = fitCurve(X, Y)

    plotConfounder(X, Y, T_hat, "Confounder after fitting curve (regression + minimizing l2 distance)")
    Nx_hat = X - s1_hat.predict(T_hat).reshape(-1, 1)
    Ny_hat = Y - s2_hat.predict(T_hat).reshape(-1, 1)

    for _ in range(K):
        T_hat = projection(T_hat, s1_hat, s2_hat, X, Y)

        plotConfounder(X, Y, T_hat, "Confounder after minimizing dependence between confounder and residuals")

        # Compute residuals
        Nx_hat = X - s1_hat.predict(T_hat).reshape(-1, 1)
        Ny_hat = Y - s2_hat.predict(T_hat).reshape(-1, 1)

        plotResiduals(Nx_hat, Ny_hat, T_hat)

        if areIndependent(T_hat, Nx_hat, Ny_hat):
            return [T_hat, s1_hat, s2_hat, np.var(Nx_hat) / np.var(Ny_hat), True]

        s1_hat, s2_hat = regressionGPR(T_hat, X, Y, Nx_hat, Ny_hat)

    return [T_hat, s1_hat, s2_hat, -1, False]  # no CAN-model fitted

def check_model(X, Y):
    # Gaussian Process Regression of Y onto X
    kernel = 1.0 * ExpSineSquared(1.0, 5.0) + WhiteKernel(1e-1)
    
    model = GaussianProcessRegressor(kernel=kernel)
    model.fit(X, Y)
    
    residuals = Y - model.predict(X)
    
    # Check for dependent residuals using HSIC
    testStat, thresh, p = hsic_gam(X, residuals, alph=0.05)

    if testStat < thresh:
        return True	# independent residuals
    else:
    	return False	# dependent residuals

# Distinguish between X->Y, Y->X, X<-T->Y, no CAN model
def causal_inference(X, Y, threshold=3):
    # Run ICAN
    T_hat, s1_hat, s2_hat, var, result = identify_confounders(X, Y)
    if (result == False):
    	return T_hat, var, result, "No CAN Model"
    
    # Decide causal structure based on variance
    if var < (1/threshold):
    	return T_hat, var, s1_hat, s2_hat, result, "X->Y"
    elif var > threshold:
    	return T_hat, var, s1_hat, s2_hat, result, "Y->X"
    else:
    	return T_hat, var, s1_hat, s2_hat, result, "X<-T->Y"
    
    # Currently not used
    # Check if X->Y or Y->X can be rejected
    modelXY = check_model(X, Y)
    modelYX = check_model(Y, X)
    
    if (not modelXY) and (not modelYX):
    	return T_hat, var, s1_hat, s2_hat, result, "X<-T->Y"
    elif modelXY and var < (1/threshold):
    	return T_hat, var, s1_hat, s2_hat, result, "X->Y"
    elif modelYX and var > threshold:
    	return T_hat, var, s1_hat, s2_hat, result, "Y->X"
    else:
    	return T_hat, var, s1_hat, s2_hat, result, "X<-T->Y"

# Start the algorithm with parameters
if len(sys.argv) != 3:
    print("Usage: python3 ican_visualized number_datapoints [dataset] \r\n [dataset] is in range [0,2]")
else:
    n = int(sys.argv[1])  # Number of datapoints
    if n < 10 or n > 1000:
    	print("Value too small or too high\r\nStarting ican with 50 data points")
    	n = 50

    # Choose dataset
    if len(sys.argv) == 2:
        dataset = 0  # default
    else:
        dataset = int(sys.argv[2]) if 0 <= int(sys.argv[2]) <= 4 else 0  # Set dataset to parameter

    T, X, Y = generate_data(n, dataset)

    plotData(X, Y, "Observed data")
    plotConfounder(X, Y, T, "True confounder (before ICAN is executed)")
    
    T_hat, var, s1_hat, s2_hat, result, structure = causal_inference(X.reshape(-1, 1), Y.reshape(-1, 1))
    
    print(f"Variance: {var}")
    print(f"Causal Structure: {structure}")
    
    if dataset == 3:
    	plotProjections(X, Y, T, T_hat)
    plotData(T, T_hat, "True confounder plotted against estimated confounder", labelX="T", labelY="estimated T")

    pdf_pages.close()

