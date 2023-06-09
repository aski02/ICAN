import numpy as np
import unittest
from ican import causal_inference

# These tests run very long (20-60 min each)
class TestICAN(unittest.TestCase):
    def test_CAN(self):
        '''
        This test checks if the algorithm correctly concludes that a CAN model can be fitted 
        ! It doesnt check X->Y, Y->X, X<-T->Y yet !

        The data is representing the following CAN-model:
        X = u(T) + Nx
        Y = v(T) + Ny

        u,v: non-linear functions
        Nx, Ny: uniformely distributed in [-0.035, 0.035]
        '''
        
        n = 100     # number of data points

        T = np.linspace(0.1, 1, n).reshape(-1,1)
        Nx = np.random.uniform(-0.035, 0.035, n).reshape(-1,1)
        Ny = np.random.uniform(-0.035, 0.035, n).reshape(-1,1)
        X = T * np.log(T) + Nx
        Y = np.square(T) + Ny

        _, _, _, _, result, _ = causal_inference(X.reshape(-1,1), Y.reshape(-1,1))
        self.assertEqual(True, result)

    def test_no_CAN(self):
        '''
        This test checks if the algorithm correctly concludes that no CAN model can be fitted

        X = u(T) + Nx
        Y = v(T) + Ny

        u,v: non-linear functions
        Nx, Ny: dependent on T
        '''
        
        n = 120     # number of data points
        
        T = np.linspace(0.1, 1, n)
        Nx = 5 * T * np.random.uniform(-0.15, 0.15, size=n)
        Ny = 8 * T * np.random.uniform(-0.15, 0.15, size=n)
        
        X = 4 * (T+1.2)**3 + 0.1 * T + Nx
        Y = 3.3 * (T-0.5)**2 + 0.3 * T + Ny
        
        _, _, _, _, result, _ = causal_inference(X.reshape(-1,1), Y.reshape(-1,1))
        self.assertEqual(False, result)
        
if __name__ == "main":
    unittest.main() 
