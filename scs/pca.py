from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def pca(data: np.ndarray) -> tuple[np.ndarray, np.array, list[float]]:
    """Principal Component Analysis.

    Uses the covariance method to project the data onto its principal components.
    Can be used to reduce the dimensionalityu of the data while trading of some
    of the explanatory power. The returned list of the explanatory power can be
    used to select how many principal components to use for dimensionality reduction.

    Args:
        data: Numpy array containing the data ordered along axis 0.

    Returns:
        The data represented in the new basis spanned by the sorted eigenvectors
            where each column of the matrix is the projection of the data onto the
            principal component corresponding to the columns index+1.
        The sorted eigenbasis on which the returned data is represented.
        A list containing the cumulative explanatory power of the all principal
            components up to the respective index. Can be used to to decide how
            many of the principal components to to select.
    """
    n = data.shape[0]
    column_mean = np.mean(data, axis=0)
    mean_subtracted_data = data - column_mean
    # only for real numbers because of simple transpose
    cov_matrix = 1 / (n - 1) * mean_subtracted_data.T @ mean_subtracted_data
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # sort eigenvalues and their corresponding eigenvectors in decreasing order
    # of the eigenvalues
    sorting_indices = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sorting_indices]
    sorted_eigenbasis = eigenvectors[:, sorting_indices]

    # project data on the eigenbasis
    projected_data = mean_subtracted_data @ sorted_eigenbasis
    pcs_explanatory_value = eigenvalues.copy()

    for i in range(1, eigenvalues.shape[0]):
        pcs_explanatory_value[i] += pcs_explanatory_value[i - 1]
    pcs_explanatory_value /= pcs_explanatory_value[-1]

    return projected_data, sorted_eigenbasis, pcs_explanatory_value


if __name__ == "__main__":
    rng = np.random.default_rng()
    data = rng.random((500, 3))
    data[:, 0] += np.linspace(0, 100, 500)
    data[:, 1] += rng.random(500) * 20 + np.linspace(0, 5, 500) ** 2
    data[:, 2] += rng.random(500) * 10 + np.linspace(0, 5, 500) ** 3
    p_data, basis, pcs_value = pca(data)
    print(basis)
    print(pcs_value)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    plt.show()

    plt.scatter(data[:, 0], data[:, 1], label="original data")
    plt.scatter(p_data[:, 0], p_data[:, 1], label="projected data")
    plt.legend()
    plt.title("First two dimensios")
    plt.show()
