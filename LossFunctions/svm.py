import numpy as np


class MultiClassSvm:
    def __init__(self, hyper_parameters):
        if not isinstance(hyper_parameters, MultiClassSvmHyperParameters):
            raise Exception("Hyper-parameter needs to be of type MultiClassSvmHyperParameters")
        self.delta = hyper_parameters.get_delta()
        self.reg_func = hyper_parameters.get_reg_func()

    def get_example_loss(self, score_vector: np.array, answer_index: np.array) -> int:
        """
        Given the score vector resulting from a data example, and the index of the correct class for that example,
        returns the score loss resulting from that example

        :param score_vector: A vector of scores
        :type score_vector: numpy.array
        :param answer_index: A vector that contains a 1 at the index of the correct answer and 0 everywhere else
        :type answer_index: numpy.array
        """
        correct_class_score = score_vector[answer_index]
        element_loss = np.maximum(0, score_vector - correct_class_score + self.delta)
        element_loss[answer_index] = 0
        return element_loss.sum()

    def get_score_loss(self, score_vectors, answer_vector):
        flipped_scores = score_vectors.transpose()
        score_loss = 0
        for i in range(answer_vector.shape[0]):
            score_loss += self.get_example_loss(flipped_scores[i], answer_vector[i])
        return score_loss


class MultiClassSvmHyperParameters:
    """
    The hyper-parameters for an svm, which default to reasonable defaults
    """
    def __init__(self):
        self.delta = 1
        self.reg_func = SvmRegularization.get_l2()

    def set_delta(self, delta):
        self.delta = delta

    def set_reg_func(self, reg_func):
        self.reg_func = reg_func

    def get_delta(self):
        return self.delta

    def get_reg_func(self):
        return self.reg_func


class SvmRegularization:
    @staticmethod
    def get_l2():
        """
        Returns an L2 regularization function

        :return: An L2 regularization function
        :rtype: A function
        """
        return L2Regularization.regularize


class L2Regularization:
    @staticmethod
    def regularize(weights: np.matrix) -> int:
        """
        Takes in a matrix of weights and computers the squared sum of all the values in it

        :param weights: A matrix of weights
        :type weights: numpy.matrix
        :return: The weighted sum of all the values in the matrix
        :rtype: int
        """
        squared_weights = np.square(weights)
        return np.sum(squared_weights)