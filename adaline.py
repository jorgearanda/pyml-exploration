# From Raschka's "Python Machine Learning" book
import numpy as np


class AdalineGD:
    """ADAptive LInear Neuron classifier

    Parameters
    ----------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Passes over the training dataset

    Attributes
    ----------
    w_: 1d-array
        Weights after fitting
    cost_: list
        Cost of misclassifications in every epoch

    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data

        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features
        y: array-like, shape = [n_samples]
            Target values

        Returns
        -------
        self: object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            output = self._net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def predict(self, X):
        """Classify sample X (as 1 or -1)"""
        return np.where(self._activation(X) >= 0.0, 1, -1)

    def _net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def _activation(self, X):
        """Compute linear activation"""
        return self._net_input(X)
