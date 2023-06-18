import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.manifold import Isomap
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from matplotlib.backends.backend_pdf import PdfPages
from hsic import hsic_gam
from datasets import generate_data

# Create pdf
pdf_pages = PdfPages('visualized.pdf')

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

    kernel = DotProduct() + WhiteKernel()  # Kernel used in example from scikit-learn documentation for GPR
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

    testStat, thresh = hsic_gam(X1, X2, alph=0.05)

    return testStat < thresh  # is true if X1, X2 are independent

def areIndependent(T_hat, Nx_hat, Ny_hat):
    return dep(Nx_hat, Ny_hat) and dep(Nx_hat, T_hat) and dep(Ny_hat, T_hat)

def projection(T_hat, s1_hat, s2_hat, X, Y):
    def depSum(T):
        T = T.reshape(-1, 1)

        Nx_hat = X - s1_hat.predict(T).reshape(-1, 1)
        Ny_hat = Y - s2_hat.predict(T).reshape(-1, 1)

        testStat1, thresh1 = hsic_gam(Nx_hat, Ny_hat, alph=0.05)
        testStat2, thresh2 = hsic_gam(Nx_hat, T, alph=0.05)
        testStat3, thresh3 = hsic_gam(Ny_hat, T, alph=0.05)

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
    kernel = DotProduct() + WhiteKernel()

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
def identify_confounders(X, Y, K=10):  # paper used K = 5000 (but if successful then termination usually occurs within 1-2 iterations)
    s1_hat, s2_hat, T_hat = fitCurve(X, Y)

    plotConfounder(X, Y, T_hat, "Confounder after fitting curve (regression + minimizing l2 distance)")

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

# Distinguish between X->Y, Y->X, X<-T->Y, no CAN model
def causal_inference(X, Y):
    # Run ICAN for X,Y and Y,X
    T_hat_XY, s1_hat_XY, s2_hat_XY, var_XY, result_XY = identify_confounders(X, Y)
    T_hat_YX, s1_hat_YX, s2_hat_YX, var_YX, result_YX = identify_confounders(Y, X)

    # Identify causal structure  (not 100% correct yet!)
    if result_XY and not result_YX:
        if var_XY < 1:  # more fine tuning needed => in the paper: var(x)/var(y) >> 1
            return [T_hat_XY, var_XY, result_XY, "X->Y"]
        else:
            return [T_hat_XY, var_XY, result_XY, "X<-T->Y"]
    elif not result_XY and result_YX:
        if var_YX < 1:
            return [T_hat_YX, var_YX, result_YX, "Y->X"]
        else:
            return [T_hat_YX, var_YX, result_YX, "X<-T->Y"]
    elif result_XY and result_YX:
        if var_XY > 1.2:
            return [T_hat_XY, var_XY, result_XY, "Y->X"]
        elif var_YX > 1.2:
            return [T_hat_XY, var_XY, result_XY, "X->Y"]
        else:
            return [T_hat_XY, var_XY, result_XY, "X<-T->Y"]
    else:
        return [T_hat_XY, var_XY, result_XY, "no CAN model"]

# Start the algorithm with parameters
if len(sys.argv) != 3:
    print("Usage: python3 ican_visualized number_datapoints [dataset] \r\n [dataset] is in range [0,2] :: default is 0")
else:
    n = int(sys.argv[1])  # Number of datapoints
    if n < 10 or n > 1000:
        n = 50

    # Choose dataset
    if len(sys.argv) == 2:
        dataset = 0  # default
    else:
        dataset = int(sys.argv[2]) if 0 <= int(sys.argv[2]) <= 4 else 0  # Set dataset to parameter

    T, X, Y = generate_data(n, dataset)

    plotData(X, Y, "Observed data")
    plotConfounder(X, Y, T, "True confounder (before ICAN is executed)")

    T_hat, var, result, structure = causal_inference(X.reshape(-1, 1), Y.reshape(-1, 1))
    print(f"Variance: {var}")

    print(f"Causal Structure: {structure}")

    plotData(T, T_hat, "True confounder plotted against estimated confounder", labelX="T", labelY="estimated T")

    pdf_pages.close()

