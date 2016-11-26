import unittest
import LossFunctions.svm as svm
import numpy as np


class TestSvm(unittest.TestCase):
    def setUp(self):
        hyper_parameters = svm.MultiClassSvmHyperParameters()
        self.svm = svm.MultiClassSvm(hyper_parameters)


class TestGetScoreFunction(TestSvm):
    def runTest(self):
        self.test_correct_example_score()
        self.test_wrong_example_score()

    def test_correct_example_score(self):
        score = np.array([2, 0, 0])
        answer = 0
        self.assertEqual(self.svm.get_example_loss(score, answer), 0, "Inaccurate for correct score")

    def test_wrong_example_score(self):
        score = np.array([1, 2, 0])
        answer = 0
        self.assertEqual(self.svm.get_example_loss(score, answer), 2, "Inaccurate for incorrect score")


class TestTotalScoreFunction(TestSvm):
    def runTest(self):
        self.test_correct_example1()

    def test_correct_example1(self):
        score = np.array([[1, 2], [10, 4], [5, 6]])
        answer = np.array([1, 2])
        self.assertEqual(self.svm.get_score_loss(score, answer), 0, "Inaccurate for correct score")

    def test_correct_example2(self):
        score = np.array([[1, 2], [10, 4], [5, 6]])
        answer = np.array([2, 2])
        self.assertNotEquals(self.svm.get_score_loss(score, answer), 0, "Inaccurate for correct score")


class TestL2Regularization(TestSvm):
    def runTest(self):
        self.test_values()

    def test_values(self):
        weights = np.matrix([[1, 2], [3, 4]])
        squared_sum = svm.L2Regularization.regularize(weights)
        self.assertEqual(squared_sum, 30, "L2 regularization returns incorrect value")
