"""A simple one-dimensional OLS estimator class.

A demo OLS estimator.

    Typical usage example:

    foo = sOLS(sample_data)
    estimation_parameters, SSR = foo.estimate_model(degree=2, bias=False)

    foo.plot_model()

    If x_data is a list or an array:
    estimation = foo.predict(x_data)

    If x is a single value, input as a list:
    estimation = foo.predict([x])
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class sOLS:
    """Simple 1-D OLS estimator.

    Plain OLS estimator for a 1-dimensional estimation.
    Allows to estimate models with parameters passed to the parameter
    estimation method.

    Attributes:
        y_true: A 1-D ndarray which contains the data points.
        y_hat: The latest prediction.
        beta: The estimation parameters.
        degree: The polynomial order of the model.
        bias: Whether the model estimation uses a bias or not.
    """

    def __init__(self, y_true: list | np.ndarray) -> None:
        """Init sOLS with args"""
        self.y_true: np.ndarray = self._data_pre_processing(y_true)
        self.y_hat: np.ndarray = np.zeros(max(self.y_true.shape))
        self.beta: np.ndarray = np.array(0)
        self.degree: int = 0
        self.bias: bool = True

    def _data_pre_processing(self, inp: list | np.ndarray) -> np.ndarray:
        """Pre-processes inpupt data.

        Ensures the dimensions and type match the requirements for usage in the
        parameter estimation.

        Args:
            inp: The one-dimensional input data.

        Returns:
            A ndarray of shape (n, 1) which can be used in subsequent matrix
            operations.

        Raises:
            TypeError: Error occurring when imp is neither a list nor an ndarray
            ValueError: Error occurring when the dimensions of inp are not equivalent
                to one-dimensional data.
        """
        if not isinstance(inp, (list, np.ndarray)):
            raise TypeError(
                f"inp must be of type List or ndarray; "
                f"received type {type(inp)} can not be processed"
            )
        inp_array = np.asarray(inp)
        if (
            len(inp_array.shape) > 2
            or len(inp_array.shape) == 2
            and min(inp_array.shape) > 1
        ):
            raise ValueError(
                f"inp must of of shape (n,), (n, 1), or (1, n); "
                f"received input shape {inp_array.shape}"
            )
        if not inp_array.shape == (max(inp_array.shape), 1):
            inp_array = inp_array.reshape(max(inp_array.shape), 1)
        return inp_array

    def _prediction_pre_processing(self, x_raw: np.ndarray) -> np.ndarray:
        """Prepares the estimaten x-value matrix.

        Generates the x-values for the model estimation based on the chosen
        degree of the model. If estimation with a bias is chosen the first column
        of the estimation x-matrix is a column of ones. Each subsequent column
        is filled with the x-values of the polynomial order correspodning to the
        column index.

        Args:
            x_raw: ndarray listing the integers [0, n]

        Returns:
            The prepared x-value matrix containing x-values corresponding to the
            chosen polynomial order. The first column is a column of ones if
            a bias is choesn for the model estimation.

        """
        bias_dim = 0
        if self.bias:
            bias_dim = 1
        x = np.zeros((x_raw.shape[0], self.degree + bias_dim))
        for d in range(self.degree + bias_dim):
            # Starts with exponent of zero if bias is included
            x[:, d] = x_raw ** (d + 1 - bias_dim)
        return x

    def estimate_model(
        self, degree: int = 0, bias: bool = True
    ) -> tuple[np.ndarray, float]:
        """Obtains beta parameter.

        Obtains the estimation parameter for a 1-dimensional estimation based on
        the passed polynomial degree using the ordinary least squares method.

        Args:
            degree: The polynomial order of the model.
            bias: Whether the model estimation uses a bias or not.

        Returns:
            A tuple which contains a ndarray with the model paramentes on the
            first and the sum of squared resiudlas on the second index.
        """
        self.degree, self.bias = degree, bias
        X = self._prediction_pre_processing(np.arange(self.y_true.shape[0]))
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ self.y_true
        self.y_hat = X @ self.beta
        eps = self.y_true - self.y_hat
        return self.beta, (eps.T @ eps)[0, 0]

    def predict(self, inp: np.ndarray | list) -> np.ndarray:
        """Predict values for given input.

        Predict the y_hat values based on the given input and the estimated model
        as avaibale in self.

        Args:
            inp: Input data on which to precict the y_hat values. Input should be
                a list or an ndarray. A single value should be passed as a list
                with a single entry e.g. [x]

        Returns:
            The prediction based on the input data in form of a one dimensional
            numpy array.

        Raises:
            ValueError: If no model has been estimated prior to calling the
                predict method.
        """
        if not self.beta:
            raise ValueError("No beta exists; Estimate model first")
        X = self._prediction_pre_processing(np.array(inp))
        self.y_hat = X @ self.beta
        return self.y_hat[:, 0]

    def plot_model(self) -> None:
        """Prints the model.

        Prints the model in a simple pyplot graph in form of a red line on top
        of the scattered true values.

        Raises:
            ValueError: If no model has been estimated prior to calling the
                plot_model method.
        """
        if self.beta is None:
            raise ValueError("No model to print; beta attribute is 'None")
        x_axis = np.arange(self.y_true.shape[0])
        fig, ax = plt.subplots()
        ax.scatter(x_axis, self.y_true, color="blue")
        ax.plot(x_axis, self.y_hat, color="red")

        title = f"Model: Polynomial of Degree {self.degree}"
        if self.bias:
            title += " with Bias"
        ax.set_title(title)
        ax.set_xlabel("Input")
        ax.set_ylabel("Y")

        ax.autoscale_view()
        plt.show()


if __name__ == "__main__":
    """Runs and estimates a demonstration model"""
    test_data = np.random.rand(20) + np.arange(20) ** 1.4
    test = sOLS(test_data)
    beta, ssr = test.estimate_model(2)
