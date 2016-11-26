import numpy as np


class MultiClassSVM:
    def __init__(self, delta, reg_func):
        """

        :param delta:
        :type delta:
        :param reg_func:
        :type reg_func:
        """
        self.delta = delta
        self.regFunc = reg_func

    @staticmethod
    def get_example_score(score_vector, correct_vector):
        """

        :param score_vector: A vector of scores
        :type score_vector: numpy.array
        :param correct_vector: A vector that contains a 1 at the index of the correct answer and 0 everywhere else
        :type correct_vector: numpy.array
        """
        for i in range(len(score_vector)):
            if correct_vector[i] == 1:
                continue

    @staticmethod
    def __get_index_of_correct_answer__(correct_vector: np.array) -> int:
        """

        :param correct_vector: A vector corresponding to the correct answer for an example, with 1 at that
        index and 0 everywhere else
        :type correct_vector: numpy.array
        """
        if correct_vector.sum() != 1:
            raise Exception("Correct vector can only contain a single 1 to indicate correct class")
        return 2

svm = MultiClassSVM(1, 2)
svm.__get_index_of_correct_answer__(np.array([1, 0, 0]))
