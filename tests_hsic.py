import unittest
import numpy as np
import pandas as pd
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
from hsic import hsic_gam

pandas2ri.activate()

def generate_data(seed, n_samples=100):
    np.random.seed(seed)
    X = np.random.randn(n_samples, 1)
    Y = X * np.random.randn(n_samples, 1) + np.random.randn(n_samples, 1)
    return X, Y

datasets = [generate_data(seed=i) for i in range(20)]

def run_python_hsic(X, Y):
    test_stat, threshold, p_value = hsic_gam(X, Y, 0.05)
    return {
        "independent": test_stat < threshold,
        "test_stat": test_stat,
        "threshold": threshold,
        "p_value": p_value
    }

def run_r_hsic(X, Y):
    X_df = pd.DataFrame(X)
    Y_df = pd.DataFrame(Y)

    # Convert pandas dataframes to R dataframes
    r_X = pandas2ri.py2rpy(X_df)
    r_Y = pandas2ri.py2rpy(Y_df)
    
    dHSIC = importr("dHSIC")
    result = r("""
    function(X, Y) {
        test_result = dhsic.test(X, Y, method = "gamma", kernel = "gaussian", alpha = 0.05)
        list(
            independent = test_result$statistic < test_result$crit.value,
            test_stat = test_result$statistic,
            threshold = test_result$crit.value,
            p_value = test_result$p.value
        )
    }
    """)(r_X, r_Y)
    
    return {
        "independent": result[0][0],
        "test_stat": result[1][0],
        "threshold": result[2][0],
        "p_value": result[3][0]
    }

class TestHSIC(unittest.TestCase):
    def test_hsic_implementation(self):
        for idx, (X, Y) in enumerate(datasets):
            python_result = run_python_hsic(X, Y)
            r_result = run_r_hsic(X, Y)
            print(python_result, r_result)
            self.assertEqual(python_result["independent"], r_result["independent"])
            self.assertAlmostEqual(python_result["test_stat"], r_result["test_stat"], places=1)
            self.assertAlmostEqual(python_result["threshold"], r_result["threshold"], places=1)
            self.assertAlmostEqual(python_result["p_value"], r_result["p_value"], places=1)

if __name__ == "__main__":
    unittest.main()
