# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------

import numpy as np
import xgboost as xgb
from sklearn.manifold import Isomap, TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA, KernelPCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import NuSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, DotProduct, RationalQuadratic, ExpSineSquared
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.optimize import minimize
from hsic import hsic_gam

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------

DIM_RED_METHODS = {
    "Isomap": Isomap,
    "TSNE": TSNE,
    "LLE": LocallyLinearEmbedding,
    "PCA": PCA,
    "KernelPCA": KernelPCA
}

KERNELS = {
    "RBF": RBF(length_scale_bounds="fixed"),
    "RBF + White": RBF(length_scale_bounds="fixed") + WhiteKernel(),
    "Matern": Matern(length_scale_bounds="fixed"),
    "Matern + White": Matern(length_scale_bounds="fixed") + WhiteKernel(),
    "Matern + ExpSineSquared": Matern(length_scale_bounds="fixed") + ExpSineSquared(length_scale_bounds="fixed"),
    "DotProduct": DotProduct(),
    "RationalQuadratic": RationalQuadratic(),
    "RationalQuadratic + White": RationalQuadratic() + WhiteKernel(),
    "ExpSineSquared": ExpSineSquared(length_scale_bounds="fixed"),
    "ExpSineSquared + White": ExpSineSquared(length_scale_bounds="fixed") + WhiteKernel()
}

REGRESSION_METHODS = {
    "GPR": GaussianProcessRegressor,
    "DecisionTree": DecisionTreeRegressor,
    "RandomForest": RandomForestRegressor,
    "NuSVR": NuSVR,
    "XGBoost": xgb.XGBRegressor
}

DEPENDENCE_MEASURES = {
    "HSIC": hsic_gam,
    "MI": mutual_info_regression,
    "Pearson": pearsonr,
    "Spearman": spearmanr,
    "Kendalltau": kendalltau
}

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def dimReduction(X, Y, method, neighbor_percentage):
    """
    Perform dimensionality reduction on the input variables X and Y

    Parameters:
    - X,Y: input variables
    - method: dimensionality reduction method
    - neighbor_percentage: specifying the neighbors parameter (only for Isomap and LLE)

    Returns:
    - T_hat: first guess for a potential confounder
    """
    data = np.column_stack((X, Y))
    
    Model = DIM_RED_METHODS[method]
    if method in ["Isomap", "LLE"]:
        neighbors = max(int(len(X) * neighbor_percentage), 2)
        model = Model(n_components=1, n_neighbors=neighbors)
    elif method == "TSNE":
        perplexity = min(30, len(X) - 1)
        model = TSNE(n_components=1, perplexity=perplexity)
    elif method == "KernelPCA":
        model = Model(n_components=1, kernel="rbf")
    else:
        model = Model(n_components=1)
     
    T_hat = model.fit_transform(data)
            
    return T_hat

def fitCurve(X, Y, dim_reduction, neighbor_percentage, regression_method, kernel, min_distance, iterations=20):
    """
    Fit a curve to the data

    Parameters:
    - X,Y: input variables
    - dim_reduction: dimensionality reduction method
    - neighbor_percentage: specifying the neighbors parameter for Isomap and LLE
    - regression_method: regression method
    - kernel: kernel used for Gaussian Process Regression (only if GPR was selected as regression method)
    - min_distance: minimization method
    - iterations: maximum number of iterations for iterative dimensionality reduction

    Returns:
    - s1_hat: Model that estimates the relationship between T and X
    - s2_hat: Model that estimates the relationship between T and Y
    - T_hat: estimated confounder
    """
    T_hat = dimReduction(X=X, Y=Y, method=dim_reduction, neighbor_percentage=neighbor_percentage)

    for _ in range(iterations):
        # Step 1: Estimate s_hat by regression
        s1_hat, s2_hat = regression(T_hat=T_hat, X=X, Y=Y, Nx_hat=0.0, Ny_hat=0.0, method=regression_method, kernel=kernel)
        s1_hat.fit(T_hat, X)
        s2_hat.fit(T_hat, Y)
        
        # Step 2: Update T_hat such that l2-distance is minimized
        def l2dist(T):
            T = T.reshape(-1, 1)
            u = s1_hat.predict(T).reshape(-1, 1)
            v = s2_hat.predict(T).reshape(-1, 1)
            dist = np.linalg.norm(np.column_stack((u, v)) - np.column_stack((X, Y)), axis=1)
            return np.sum(dist)

        # Minimize l2-distance w.r.t. T_hat
        init_guess = T_hat.flatten()
        T_hat_new = minimize(l2dist, init_guess, method=min_distance).x.reshape(-1, 1)

        # Check for convergence
        if np.linalg.norm(T_hat_new - T_hat) / np.linalg.norm(T_hat) < 1e-3:
            break

        T_hat = T_hat_new
        
    return s1_hat, s2_hat, T_hat

def compute_dependence(X1, X2, method, threshold):
    """
    Measures dependence between X1 and X2

    Parameters:
    - X1,X2: input variables
    - method: dependence measure
    - threshold: threshold for determining if variables are independent or dependent

    Returns:
    - if method is Hilbert-Schmidt Independence Criterion: testStat, thresh, p-value
    - if method is Mutual Information: estimated mutual information
    - if method is Pearson, Spearman or Kendalltau: correlation coefficient
    """
    if method == "HSIC":
        return DEPENDENCE_MEASURES[method](X1, X2, alph=threshold)
    elif method == "MI":
        return DEPENDENCE_MEASURES[method](X1, np.squeeze(X2))
    elif method in ["Pearson", "Spearman", "Kendalltau"]:
        return abs(DEPENDENCE_MEASURES[method](X1.flatten(), X2.flatten())[0])
    else:
        raise ValueError(f"Unkown method: {method}")
    
def independent(X1, X2, method, threshold):
    """
    Checks for dependence between X1 and X2

    Parameters:
    - X1, X2: input variables
    - method: dependence measure
    - threshold: threshold for determining if variables are independent or dependent

    Returns:
    - True if X1 and X2 are independent, False otherwise
    """
    test_statistic = compute_dependence(X1=X1, X2=X2, method=method, threshold=threshold)
    
    if method == "HSIC":
        testStat, thresh, _ = test_statistic
        return testStat < thresh
    elif method == "MI":
        return test_statistic < threshold
    elif method in ["Pearson", "Spearman", "Kendalltau"]:
        return test_statistic < threshold
    else:
        raise ValueError(f"Unkown method: {method}")

def areIndependent(T_hat, Nx_hat, Ny_hat, method, threshold):
    """
    Tests if T_hat, Nx_hat and Ny_hat are independent

    Parameters:
    - T_hat: estimated confounder
    - Nx_hat: estimated noise of X
    - Ny_hat: estimated noise of Y
    - method: dependence measure
    - threshold: threshold for determining if variables are independent or dependent

    Returns:
    - True if T_hat, Nx_hat and Ny_hat are independent, False otherwise
    """
    return independent(X1=Nx_hat, X2=Ny_hat, method=method, threshold=threshold) and independent(X1=Nx_hat, X2=Ny_hat, method=method, threshold=threshold) and independent(X1=Nx_hat, X2=Ny_hat, method=method, threshold=threshold)

def projection(T_hat, s1_hat, s2_hat, X, Y, method, threshold, min_projection):
    """
    Estimate T_hat by minimizing dependence between T_hat, Nx_hat, Ny_hat

    Parameters:
    - T_hat: current estimate for the confounder
    - s1_hat: Model that estimates the relationship between T and X
    - s2_hat: Model that estimates the relationship between T and Y
    - X,Y: input variables
    - method: dependence measure
    - threshold: threshold for determining if variables are independent or dependent
    - min_projection: minimization method

    Returns:
    - T_hat: updated estimate for the confounder
    """
    def depSum(T):
        T = T.reshape(-1, 1)
        Nx_hat = X - s1_hat.predict(T).reshape(-1, 1)
        Ny_hat = Y - s2_hat.predict(T).reshape(-1, 1)

        if method == "HSIC":
        	_, _, p1 = compute_dependence(Nx_hat, Ny_hat, method, threshold)
        	_, _, p2 = compute_dependence(Nx_hat, T, method, threshold)
        	_, _, p3 = compute_dependence(Ny_hat, T, method, threshold)
        	return -(p1 + p2 + p3)
        elif method == "MI":
        	mi1 = compute_dependence(Nx_hat, np.squeeze(Ny_hat), method, threshold)
        	mi2 = compute_dependence(Nx_hat, np.squeeze(T), method, threshold)
        	mi3 = compute_dependence(Ny_hat, np.squeeze(T), method, threshold)
        	return mi1 + mi2 + mi3
        elif method in ["Pearson", "Spearman", "Kendalltau"]:
        	dep1 = compute_dependence(Nx_hat, Ny_hat, method, threshold)
        	dep2 = compute_dependence(Nx_hat, T, method, threshold)
        	dep3 = compute_dependence(Ny_hat, T, method, threshold)
        	return dep1 + dep2 + dep3
        else:
            raise ValueError(f"Unkown method: {method}")

    # Minimize dependence w.r.t. T_hat
    init_guess = T_hat.flatten()
    T_hat = minimize(depSum, init_guess, method=min_projection, options={"maxiter": 5000}).x.reshape(-1, 1)
    
    return T_hat

def regression(T_hat, X, Y, Nx_hat, Ny_hat, method, kernel):
    """
    Estimate s1_hat and s2_hat by regression: (X,Y) = (s1_hat(T_hat), s2_hat(T_hat)) + (Nx_hat, Ny_hat)

    Parameters:
    - T_hat: estimated confounder
    - X,Y: input variables
    - Nx_hat, Ny_hat: estimated noise of X and Y
    - method: regression method
    - kernel: kernel used for Gaussian Process Regression (only if GPR was selected as regression method)

    Returns:
    - s1_hat: Model that estimates the relationship between T and X
    - s2_hat: Model that estimates the relationship between T and Y
    """
    Model = REGRESSION_METHODS[method]
    s1_hat = Model(kernel=KERNELS[kernel], n_restarts_optimizer=10) if method == "GPR" else Model()
    s2_hat = Model(kernel=KERNELS[kernel], n_restarts_optimizer=10) if method == "GPR" else Model()
    
    s1_hat.fit(T_hat, (X - Nx_hat).ravel())
    s2_hat.fit(T_hat, (Y - Ny_hat).ravel())
    
    return s1_hat, s2_hat

# -------------------------------------------------------------------------
# ICAN-Algorithm
# -------------------------------------------------------------------------

def identify_confounders(X, Y, dim_reduction, neighbor_percentage, iterations, kernel, independence_threshold, regression_method, independence_method, min_distance, min_projection):
    """
    ICAN-Algorithm

    Parameters:
    - X,Y: input variables
    - dim_reduction: dimensionality reduction method
    - neighbor_percentage: specifying the neighbors parameter (only for Isomap and LLE)
    - iterations: maximum number of iterations for the main loop
    - regression_method: regression method
    - kernel: kernel used for Gaussian Process Regression (only if GPR was selected as regression method)
    - threshold: threshold for determining if variables are independent or dependent
    - independence_method: dependence measure
    - min_distance: minimization method in the fit-curve step
    - min_projection: minimization method in the projection step

    Returns:
    - T_hat: estimated confounder
    - s1_hat: Model that estimates the relationship between T and X
    - s2_hat: Model that estimates the relationship between T and Y
    - variance quotient: variance(Nx_hat) / variance(Ny_hat)
    - result: True if CAN-model could be fitted to the data, False otherwise
    """
    s1_hat, s2_hat, T_hat = fitCurve(X=X, Y=Y, dim_reduction=dim_reduction, neighbor_percentage=neighbor_percentage, regression_method=regression_method, kernel=kernel, min_distance=min_distance)
    
    for _ in range(iterations):	
        T_hat = projection(T_hat=T_hat, s1_hat=s1_hat, s2_hat=s2_hat, X=X, Y=Y, method=independence_method, threshold=independence_threshold, min_projection=min_projection)

        # Compute residuals
        Nx_hat = X - s1_hat.predict(T_hat).reshape(-1, 1)
        Ny_hat = Y - s2_hat.predict(T_hat).reshape(-1, 1)

        if areIndependent(T_hat=T_hat, Nx_hat=Nx_hat, Ny_hat=Ny_hat, method=independence_method, threshold=independence_threshold):
            return [T_hat, s1_hat, s2_hat, np.round(np.var(Nx_hat) / np.var(Ny_hat), 4), True]

        s1_hat, s2_hat = regression(T_hat=T_hat, X=X, Y=Y, Nx_hat=Nx_hat, Ny_hat=Ny_hat, method=regression_method, kernel=kernel)
    
    # Compute residuals
    Nx_hat = X - s1_hat.predict(T_hat).reshape(-1, 1)
    Ny_hat = Y - s2_hat.predict(T_hat).reshape(-1, 1)
    
    return [T_hat, s1_hat, s2_hat, np.round(np.var(Nx_hat) / np.var(Ny_hat), 4), False]

# -------------------------------------------------------------------------
# Determining causal structure
# -------------------------------------------------------------------------

def check_model(X, Y, threshold):
    """
    Check if causal direction X->Y can be accepted or rejected

    Parameters:
    - X,Y: input variables
    - threshold: threshold for determining if variables are independent or dependent

    Results:
    - result: True if causal direction X->Y was accepted, False otherwise
    - p: p-value indicating if residuals are independent or not
    """
    # Regression of Y onto X
    kernel = KERNELS["RBF"]
    model = GaussianProcessRegressor(kernel=kernel)
    model.fit(X, Y)
    
    residuals = Y.reshape(-1,1) - model.predict(X).reshape(-1,1)
        
    # Check for dependent residuals using HSIC
    testStat, thresh, p = hsic_gam(X, residuals, alph=threshold)
    
    if testStat < thresh:
        return True, np.round(p, 4)
    else:
    	return False, np.round(p, 4)	

def causal_inference(X, Y, dim_reduction, neighbor_percentage, iterations, kernel, variance_threshold, independence_threshold, regression_method, independence_method, min_distance, min_projection):
    """
    Distinguish between X->Y, Y->X, X<-T->Y, no CAN model

    Parameters:
    - X,Y: input variables
    - dim_reduction: dimensionality reduction method
    - neighbor_percentage: specifying the neighbors parameter (only for Isomap and LLE)
    - iterations: maximum number of iterations for the main loop
    - regression_method: regression method
    - kernel: kernel used for Gaussian Process Regression (only if GPR was selected as regression method)
    - threshold: threshold for determining if variables are independent or dependent
    - independence_method: dependence measure
    - min_distance: minimization method in the fit-curve step
    - min_projection: minimization method in the projection step
    - variance_threshold: threshold for determining when to distinguish between a direct relationship or a confounder

    Returns:
    - T_hat: estimated confounder
    - variance quotient: variance(Nx_hat) / variance(Ny_hat)
    - s1_hat: Model that estimates the relationship between T and X
    - s2_hat: Model that estimates the relationship between T and Y
    - result: True if CAN-model could be fitted to the data, False otherwise
    - causal structure: X->Y, Y->X, X<-T->Y, no CAN model
    - p1: p-value indicating if residuals are independent or not
    - p2: p-value indicating if residuals are independent or not
    """
    # Normalize X,Y to variance 1
    X = (X - np.mean(X)) / np.std(X)
    Y = (Y - np.mean(Y)) / np.std(Y)
    
    # Check if X and Y are independent
    testStat, thresh, _ = hsic_gam(X=X, Y=Y, alph=independence_threshold)
    if testStat < thresh:
    	return None, None, None, None, None, "No relationship", None, None
        
    # Run ICAN
    T_hat, s1_hat, s2_hat, var, result = identify_confounders(X=X, Y=Y, dim_reduction=dim_reduction, neighbor_percentage=neighbor_percentage, iterations=iterations, kernel=kernel, independence_threshold=independence_threshold, regression_method=regression_method, independence_method=independence_method, min_distance=min_distance, min_projection=min_projection)
   
    # Check if X->Y or Y->X can be rejected
    modelXY, p1 = check_model(X=X, Y=Y, threshold=independence_threshold)
    modelYX, p2 = check_model(X=Y, Y=X, threshold=independence_threshold)
    
    # Check if CAN model was accepted
    if (result == False):
        return T_hat, var, s1_hat, s2_hat, result, "No CAN Model", p1, p2
    
    # Distinguish between X->Y, Y->X and X<-T->Y
    if modelXY and var < (1/variance_threshold):
        return T_hat, var, s1_hat, s2_hat, result, "X->Y", p1, p2
    elif modelYX and var > variance_threshold:
        return T_hat, var, s1_hat, s2_hat, result, "Y->X", p1, p2
    else:
        return T_hat, var, s1_hat, s2_hat, result, "X<-T->Y", p1, p2
