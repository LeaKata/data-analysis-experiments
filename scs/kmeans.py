from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def k_means(
    data: np.ndarray, k: int, repetitions: int, max_iterations: int = 10000
) -> tuple[np.ndarray, np.ndarray, float]:
    """K-Means clustering.

    Uses a basic mean initialization of randomly picking points from the data
    as starting means. Repeats the clustering process 'repitiion' number of times
    and returns the clustering with the lowest average distance from each datapoint
    to its respective clustermean.

    Args:
        data: Numpy array containing the data with samples ordered along axis 0.
        k: Integer number of clusters.
        repetitions: Integer count of repetitions from which to choose the best
            clustering.
        max_iterations: Integer max number of update steps for each clustering
            process.

    Returns:
        A numpy array containing the best cluster indices for each datapoint index.
        The means used for the for the returned clustering.
        A float loss average squared distance of each datapoint to its respective
            cluster mean.
    """
    rng = np.random.default_rng()
    loss_hist = []
    means_hist = []
    for _ in (repetition := tqdm(range(repetitions))):
        repetition.set_description(f"{k} clusters")
        means = rng.choice(data, k, replace=False, axis=0)
        for _ in range(max_iterations):
            distances = np.linalg.norm(data[:, np.newaxis] - means, axis=2)
            labels = np.argmin(distances, axis=1)
            old_mean = means.copy()
            for mean in range(k):
                means[mean] = np.mean(data[np.argwhere(labels == mean)[:, 0]], axis=0)
            if (old_mean - means < 1e-10).all():
                break
        means_hist.append(means)
        loss_hist.append(np.sum(np.min(distances, axis=1) ** 2) / data.shape[0])
        repetition.set_postfix(last_loss=loss_hist[-1])
    best_means = means_hist[np.argmin(loss_hist)]
    labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - best_means, axis=2), axis=1)
    return labels, best_means, float(np.min(loss_hist))


if __name__ == "__main__":
    """Clusters random generated demonstration data"""
    rng = np.random.default_rng()
    data = rng.random((500, 2))
    labels, means, loss = k_means(data, 3, 100)
    print(f"Loss: {loss}\n{means}")

    for c in range(3):
        cluster = data[np.argwhere(labels == c)[:, 0]]
        plt.scatter(cluster[:, 0], cluster[:, 1])
        plt.scatter(means[c, 0], means[c, 1], marker="x", c="k")
    plt.show()
