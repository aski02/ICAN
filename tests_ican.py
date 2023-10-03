import unittest
from datasets import generate_data
from ican import causal_inference

class TestComputeAccuracy(unittest.TestCase):
    
    def setUp(self):
        self.dim_reduction = "Isomap"
        self.neighbor_percentage = 0.1
        self.iterations = 3
        self.kernel = "RBF"
        self.variance_threshold = 2.0
        self.independence_threshold = 0.05
        self.regression_method = "GPR"
        self.independence_method = "HSIC"
        self.min_distance = "Nelder-Mead"
        self.min_projection = "Nelder-Mead"

    def check_dataset(self, dataset_index, expected_structure):
        T, X, Y = generate_data(40, 6 + dataset_index)
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)

        _, _, _, _, _, structure, _, _ = causal_inference(X, Y, self.dim_reduction, self.neighbor_percentage, self.iterations, self.kernel, self.variance_threshold, self.independence_threshold, self.regression_method, self.independence_method, self.min_distance, self.min_projection)
        
        self.assertEqual(structure, expected_structure)

    def test_dataset_1(self):
        self.check_dataset(0, "X->Y")

    def test_dataset_2(self):
        self.check_dataset(1, "X<-T->Y")

    def test_dataset_3(self):
        self.check_dataset(2, "X<-T->Y")

    def test_dataset_4(self):
        self.check_dataset(3, "Y->X")

    def test_dataset_5(self):
        self.check_dataset(4, "X<-T->Y")

    def test_dataset_6(self):
        self.check_dataset(5, "Y->X")

    def test_dataset_7(self):
        self.check_dataset(6, "X->Y")

    def test_dataset_8(self):
        self.check_dataset(7, "X<-T->Y")

    def test_dataset_9(self):
        self.check_dataset(8, "X->Y")

    def test_dataset_10(self):
        self.check_dataset(9, "X<-T->Y")

if __name__ == '__main__':
    unittest.main()
