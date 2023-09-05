from sklearn.manifold import Isomap, TSNE, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.decomposition import PCA, KernelPCA
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, DotProduct, ExpSineSquared, RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from scipy.optimize import minimize
from hsic import hsic_gam
import numpy as np


# Initial dimensionality Reduction
def dimReduction(X, Y, method, neighbor_percentage):
    X = X.flatten()
    Y = Y.flatten()
    data = np.concatenate((X[:, np.newaxis], Y[:, np.newaxis]), axis=1)
    neighbors = max(int(X.shape[0] * neighbor_percentage), 2)
    
    if method == "Isomap":
    	iso = Isomap(n_components=1, n_neighbors=neighbors)
    	T_hat = iso.fit_transform(data)
    elif method == "PCA":
        pca = PCA(n_components=1)
        T_hat = pca.fit_transform(data)
    elif method == "TSNE":
        tsne = TSNE(n_components=1)
        T_hat = tsne.fit_transform(data)
    elif method == "KernelPCA":
        kpca = KernelPCA(n_components=1)
        T_hat = kpca.fit_transform(data)
    elif method == "LLE":
        lle = LocallyLinearEmbedding(n_components=1, n_neighbors=neighbors)
        T_hat = lle.fit_transform(data)
    elif method == "SpectralEmbedding":
        se = SpectralEmbedding(n_components=1, n_neighbors=neighbors)
        T_hat = se.fit_transform(data)
        
    return T_hat


def fitCurve(X, Y, dim_reduction, neighbor_percentage, regression_method, kernel, min_distance, iterations=5):
    T_hat = dimReduction(X, Y, dim_reduction, neighbor_percentage)
    
    s1_hat, s2_hat = regression(T_hat, X, Y, 0.0, 0.0, regression_method, kernel)

    for _ in range(iterations):
        # Step 1: Estimate s_hat using Gaussian Process Regression
        s1_hat.fit(T_hat, X)
        s2_hat.fit(T_hat, Y)

        # Step 2: Update T_hat such that l2-distance is minimized
        def l2dist(T):
            T = T.reshape(-1, 1)
            return np.linalg.norm([s1_hat.predict(T).reshape(-1, 1) - X, s2_hat.predict(T).reshape(-1, 1) - Y])

        # Minimize l2-distance w.r.t. T_hat
        init_guess = T_hat.flatten()
        T_hat_new = minimize(l2dist, init_guess, method=min_distance).x.reshape(-1, 1)
        
        # Check for convergence
        if np.linalg.norm(T_hat_new - T_hat) / np.linalg.norm(T_hat) < 1e-4:
            break

        T_hat = T_hat_new

    return s1_hat, s2_hat, T_hat


# Independence test
def compute_test_statistic(X1, X2, method, threshold):
    if method == "HSIC":
    	return hsic_gam(X1, X2, alph=threshold)
    elif method == "MI":
    	return mutual_info_regression(X1, np.squeeze(X2))


def dep(X1, X2, method, threshold):
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(-1, 1)
    
    test_statistic = compute_test_statistic(X1, X2, method, threshold)
    
    if method == "HSIC":
    	testStat, thresh, p = test_statistic
    	return testStat < thresh	# is true if X1, X2 are independent
    elif method == "MI":
    	mi = test_statistic
    	return mi < threshold


def areIndependent(T_hat, Nx_hat, Ny_hat, method, threshold):
    return dep(Nx_hat, Ny_hat, method, threshold) and dep(Nx_hat, T_hat, method, threshold) and dep(Ny_hat, T_hat, method, threshold)


def projection(T_hat, s1_hat, s2_hat, X, Y, method, threshold, min_projection):
    def depSum(T):
        T = T.reshape(-1, 1)

        Nx_hat = X - s1_hat.predict(T).reshape(-1, 1)
        Ny_hat = Y - s2_hat.predict(T).reshape(-1, 1)

        if method == "HSIC":
        	_, _, p1 = compute_test_statistic(Nx_hat, Ny_hat, method, threshold)
        	_, _, p2 = compute_test_statistic(Nx_hat, T, method, threshold)
        	_, _, p3 = compute_test_statistic(Ny_hat, T, method, threshold)
        	
        	return -(p1 + p2 + p3)
        elif method == "MI":
        	mi1 = compute_test_statistic(Nx_hat, np.squeeze(Ny_hat), method, threshold)
        	mi2 = compute_test_statistic(Nx_hat, np.squeeze(T), method, threshold)
        	mi3 = compute_test_statistic(Ny_hat, np.squeeze(T), method, threshold)
        	
        	return mi1 + mi2 + mi3

    # Minimize dependence w.r.t. T_hat
    init_guess = T_hat.flatten()
    T_hat = minimize(depSum, init_guess, method=min_projection).x.reshape(-1, 1)

    return T_hat


def getKernel(kernel):
    if kernel == "Matern + White":
        kernel = Matern(length_scale_bounds="fixed") + WhiteKernel()
    elif kernel == "Matern + ExpSineSquared":
        kernel = Matern(length_scale_bounds="fixed") + ExpSineSquared(length_scale_bounds="fixed")
    elif kernel == "Matern":
        kernel = Matern(length_scale_bounds="fixed")
    elif kernel == "RBF + White":
        kernel = RBF(length_scale_bounds="fixed") + WhiteKernel()
    elif kernel == "RBF":
        kernel = RBF(length_scale_bounds="fixed")
    elif kernel == "DotProduct + White":
        kernel = DotProduct() + WhiteKernel()
    elif kernel == "DotProduct":
        kernel = DotProduct()
    elif kernel == "RationalQuadratic + White":
        kernel = RationalQuadratic() + WhiteKernel()
    elif kernel == "RationalQuadratic":
        kernel = RationalQuadratic()
    elif kernel == "ExpSineSquared + White":
        kernel = ExpSineSquared(length_scale_bounds="fixed") + WhiteKernel()
    elif kernel == "ExpSineSquared":
        kernel = ExpSineSquared(length_scale_bounds="fixed")
    
    return kernel


def regression(T_hat, X, Y, Nx_hat, Ny_hat, method, kernel):
    kernel = getKernel(kernel)
    
    if method == "GPR":
        s1_hat = GaussianProcessRegressor(kernel=kernel)
        s2_hat = GaussianProcessRegressor(kernel=kernel)
    elif method == "DecisionTree":
        s1_hat = DecisionTreeRegressor()
        s2_hat = DecisionTreeRegressor()
    elif method == "RandomForest":
        s1_hat = RandomForestRegressor()
        s2_hat = RandomForestRegressor()
    
    s1_hat.fit(T_hat, X - Nx_hat)
    s2_hat.fit(T_hat, Y - Ny_hat)
    
    return s1_hat, s2_hat


# ICAN algorithm
def identify_confounders(X, Y, dim_reduction, neighbor_percentage, iterations, kernel, independence_threshold, regression_method, independence_method, min_distance, min_projection):
    s1_hat, s2_hat, T_hat = fitCurve(X, Y, dim_reduction, neighbor_percentage, regression_method, kernel, min_distance)
    
    for _ in range(iterations):	
        T_hat = projection(T_hat, s1_hat, s2_hat, X, Y, independence_method, independence_threshold, min_projection)

        # Compute residuals
        Nx_hat = X - s1_hat.predict(T_hat).reshape(-1, 1)
        Ny_hat = Y - s2_hat.predict(T_hat).reshape(-1, 1)

        if areIndependent(T_hat, Nx_hat, Ny_hat, independence_method, independence_threshold):
            return [T_hat, s1_hat, s2_hat, np.round(np.var(Nx_hat) / np.var(Ny_hat), 4), True]

        s1_hat, s2_hat = regression(T_hat, X, Y, Nx_hat, Ny_hat, regression_method, kernel)
    
    # Compute residuals
    Nx_hat = X - s1_hat.predict(T_hat).reshape(-1, 1)
    Ny_hat = Y - s2_hat.predict(T_hat).reshape(-1, 1)
    
    return [T_hat, s1_hat, s2_hat, np.round(np.var(Nx_hat) / np.var(Ny_hat), 4), False]  # no CAN-model fitted


# Check if X->Y can be rejected
def check_model(X, Y, kernel, threshold):
    # Regression of Y onto X
    kernel = getKernel(kernel)
    model = GaussianProcessRegressor(kernel=kernel)
    model.fit(X, Y)
    
    residuals = Y.reshape(-1,1) - model.predict(X).reshape(-1,1)
    
    # Check for dependent residuals using HSIC
    testStat, thresh, p = hsic_gam(X, residuals, alph=threshold)
    
    if testStat < thresh:
        return True, np.round(p[0][0], 4)
    else:
    	return False, np.round(p[0][0], 4)	


# Distinguish between X->Y, Y->X, X<-T->Y, no CAN model
def causal_inference(X, Y, dim_reduction, neighbor_percentage, iterations, kernel, variance_threshold, independence_threshold, regression_method, independence_method, min_distance, min_projection):
    # Normalize to variance 1
    X = (X - np.mean(X)) / np.std(X)
    Y = (Y - np.mean(Y)) / np.std(Y)
    
    # Run ICAN
    T_hat, s1_hat, s2_hat, var, result = identify_confounders(X, Y, dim_reduction, neighbor_percentage, iterations, kernel, independence_threshold, regression_method, independence_method, min_distance, min_projection)
   
    # Check if X->Y or Y->X can be rejected
    modelXY, p1 = check_model(X, Y, kernel, independence_threshold)
    modelYX, p2 = check_model(Y, X, kernel, independence_threshold)

    # Check if CAN model was accepted
    if (result == False):
    	return T_hat, var, s1_hat, s2_hat, result, "No CAN Model", p1, p2
    
    # Distinguish between X->Y, Y->X and X<-T->Y
    if (not modelXY) and (not modelYX):
    	return T_hat, var, s1_hat, s2_hat, result, "X<-T->Y", p1, p2
    elif modelXY and var < (1/variance_threshold):
    	return T_hat, var, s1_hat, s2_hat, result, "X->Y", p1, p2
    elif modelYX and var > variance_threshold:
    	return T_hat, var, s1_hat, s2_hat, result, "Y->X", p1, p2
    else:
    	return T_hat, var, s1_hat, s2_hat, result, "X<-T->Y", p1, p2

