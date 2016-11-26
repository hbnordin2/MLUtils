import unittest
import LossFunctions.svm as svm
import numpy as np


class TestSvm(unittest.TestCase):
    def setUp(self):
        self.svm = svm.MultiClassSVM(1, "stub")


class TestGetScoreFunction(TestSvm):
    def runTest(self):
        self.test_correct_example_score()
        self.test_wrong_example_score()

    def test_correct_example_score(self):
        score = np.array([2, 0, 0])
        answer = 0
        self.assertEqual(self.svm.get_example_loss(score, answer), 0, "Accurate for correct score")

    def test_wrong_example_score(self):
        score = np.array([1, 2, 0])
        answer = 0
        self.assertEqual(self.svm.get_example_loss(score, answer), 2, "Accurate for incorrect score")


class TestL2Regularization(TestSvm):
    def runTest(self):
        self.test_values()

    def test_values(self):
        weights = np.matrix([[1, 2], [3, 4]])
        squared_sum = svm.L2Regularization.regularize(weights)
        self.assertEqual(squared_sum, 30, "L2 regularization returns incorrect value")
