import numpy as np
import unittest
from hsic import hsic_gam

# These tests just check for the correct result not for the correct values (testStat/statistic and threshold/critical value) 
# Because the exact values are not equal I will implement hsic myself later on
class TestHSIC(unittest.TestCase):
    def test_independent(self):
        '''
        This test checks if the python implementation of HSIC correctly identifies independent data (Reference is dHSIC)
        '''

        X1 = np.array([0.03347964,  0.00060912,  0.0219115 , -0.02419677, -0.00194896, -0.00401801, -0.02516746, -0.00683531, -0.01023302, -0.02270406]).reshape(-1, 1)
        Y1 = np.array([-0.02873904, -0.00573243, -0.01116284, -0.03167527,  0.02813263, -0.0253398 ,  0.01373112,  0.0089316 , -0.03127794,  0.03083105]).reshape(-1, 1)

        X2 = np.array([0.0772463 , -0.91260841,  0.28969392,  0.71568856, -0.08272605, -0.98216155, -0.40104102, -0.70441517,  0.30185897,  0.83438841]).reshape(-1, 1)
        Y2 = np.array([-0.01700912, -0.02652175, -0.02874346,  0.02379214,  0.01087314, -0.03414935,  0.0254937 ,  0.02086221,  0.02914415, -0.01038954]).reshape(-1, 1)

        X3 = np.array([-0.04229407,  0.33559807, -0.24695701, -0.15193704, -0.13076795, -0.6931439 ,  0.21908194,  0.26909248,  0.21182721, -0.02992354]).reshape(-1, 1)
        Y3 = np.array([6.67830711,  2.7222485 ,  4.51723399,  3.94678569, -1.06820258, 0.6540868 ,  6.79006549,  1.39713812, -2.99670966,  2.43742226]).reshape(-1, 1)

        testStat1, thresh1 = hsic_gam(X1, Y1)
        testStat2, thresh2 = hsic_gam(X2, Y2)
        testStat3, thresh3 = hsic_gam(X3, Y3)

        res1 = testStat1 < thresh1
        res2 = testStat2 < thresh2
        res3 = testStat3 < thresh3

        result = np.alltrue([res1, res2, res3])
        self.assertEqual(True, result)

    def test_dependent(self):
        '''
        This test checks if the python implementation HSIC correctly identifies dependent data (Reference is dHSIC)
        '''
        X1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).reshape(-1, 1)
        Y1 = np.array([100, 80, 60, 40, 20, 0, -20, -40, -60, -80]).reshape(-1, 1)

        X2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
        Y2 = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100]).reshape(-1, 1)

        X3 = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).reshape(-1, 1)
        Y3 = np.array([4, 16, 36, 64, 100, 144, 196, 256, 324, 400]).reshape(-1, 1)

        testStat1, thresh1 = hsic_gam(X1, Y1)
        testStat2, thresh2 = hsic_gam(X2, Y2)
        testStat3, thresh3 = hsic_gam(X3, Y3)

        res1 = testStat1 > thresh1
        res2 = testStat2 > thresh2
        res3 = testStat3 > thresh3
        
        result = np.alltrue([res1, res2, res3])

        self.assertEqual(True, result)

if __name__ == "main":
    unittest.main() 