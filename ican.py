import numpy as np
from scipy.optimize import minimize
from sklearn.manifold import Isomap
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ExpSineSquared
from sklearn.neighbors import NearestNeighbors
from hsic import hsic_gam

# Initial dimensionality Reduction using Isomap
def dimReduction(X, Y):
    X = X.flatten()
    Y = Y.flatten()
    data = np.concatenate((X[:, np.newaxis], Y[:, np.newaxis]), axis=1)  # Isomap requires 2 columns (samples x dimensions)

    neighbors = int(X.shape[0] * 0.1)  # high value: more neighbors are taken into account => smoother curve (less complexity)

    iso = Isomap(n_components=1, n_neighbors=neighbors)  # n_components = 1 because T_hat is 1-dimensional
    iso.fit(data)
    T_hat = iso.transform(data)

    return T_hat

def fitCurve(X, Y):
    # Initial dimensionality reduction
    T_hat = dimReduction(X, Y)
    
    # Define kernel for Gaussian Process Regression
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-2, 0.5))
    s1_hat = GaussianProcessRegressor(kernel=kernel, alpha=1e-1, normalize_y=True, n_restarts_optimizer=10)
    s2_hat = GaussianProcessRegressor(kernel=kernel, alpha=1e-1, normalize_y=True, n_restarts_optimizer=10)

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

# Independence test using Hilbert-Schmidt Independence Criterion
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
    T_hat = minimize(depSum, init_guess, method="Nelder-Mead").x.reshape(-1, 1)

    return T_hat

# Regression using Gaussian Processes
def regressionGPR(T_hat, X, Y, Nx_hat, Ny_hat):
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-2, 0.5))
    
    s1_hat = GaussianProcessRegressor(kernel=kernel, alpha=1e-1, normalize_y=True, n_restarts_optimizer=10)
    s2_hat = GaussianProcessRegressor(kernel=kernel, alpha=1e-1, normalize_y=True, n_restarts_optimizer=10)

    s1_hat.fit(T_hat, X - Nx_hat)
    s2_hat.fit(T_hat, Y - Ny_hat)

    return s1_hat, s2_hat

# ICAN algorithm
def identify_confounders(X, Y, K=4):
    s1_hat, s2_hat, T_hat = fitCurve(X, Y)

    for _ in range(K):
        T_hat = projection(T_hat, s1_hat, s2_hat, X, Y)

        # Compute residuals
        Nx_hat = X - s1_hat.predict(T_hat).reshape(-1, 1)
        Ny_hat = Y - s2_hat.predict(T_hat).reshape(-1, 1)

        if areIndependent(T_hat, Nx_hat, Ny_hat):
            return [T_hat, s1_hat, s2_hat, np.var(Nx_hat) / np.var(Ny_hat), True]

        s1_hat, s2_hat = regressionGPR(T_hat, X, Y, Nx_hat, Ny_hat)

    return [T_hat, s1_hat, s2_hat, -1, False]  # no CAN-model fitted

# Check if X->Y can be rejected
def check_model(X, Y):
    # Gaussian Process Regression of Y onto X
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-2, 1e2))
    model = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True, n_restarts_optimizer=10)
    
    model.fit(X, Y)
    residuals = Y.reshape(-1,1) - model.predict(X).reshape(-1,1)
    
    # Check for dependent residuals using HSIC
    testStat, thresh, p = hsic_gam(X.reshape(-1,1), residuals.reshape(-1,1), alph=0.05)

    if testStat < thresh:
        return True	# independent residuals
    else:
    	return False	# dependent residuals

# Distinguish between X->Y, Y->X, X<-T->Y, no CAN model
def causal_inference(X, Y, threshold=1.5):
    # Run ICAN
    T_hat, s1_hat, s2_hat, var, result = identify_confounders(X, Y)
    if (result == False):
    	return T_hat, var, s1_hat, s1_hat, result, "No CAN Model"
   
    # Check if X->Y or Y->X can be rejected
    modelXY = check_model(X, Y)
    modelYX = check_model(Y, X)
    
    # Distinguish between X->Y, Y->X and X<-T->Y
    if (not modelXY) and (not modelYX):
    	return T_hat, var, s1_hat, s2_hat, result, "X<-T->Y"
    elif modelXY and modelYX:
    	return T_hat, var, s1_hat, s2_hat, result, "both directions possible"
    elif modelXY and var < (1/threshold):
    	return T_hat, var, s1_hat, s2_hat, result, "X->Y"
    elif modelYX and var > threshold:
    	return T_hat, var, s1_hat, s2_hat, result, "Y->X"
    else:
    	return T_hat, var, s1_hat, s2_hat, result, "X<-T->Y"

