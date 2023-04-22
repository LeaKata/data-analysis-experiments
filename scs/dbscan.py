from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def _get_neighbors(data: np.ndarray, point: np.ndarray, epsilon: float) -> list[int]:
    """Get all neighbors around point.

    Applies a distance measure to all datapoints and returns all datapoints which
    are closer than epsilon in a list of neighbors. Can use any distance measure of
    choice - here we use the Euclidean norm as a default.

    Args:
        data: Numpy array containing the data with samples ordered along axis 0.
        point: The point for which to return the neighbours.
        epsilon: Max distance at which a datapoint is considered to be in a
            neighbor of the passed point.

    Returns:
        A list of the indices of the datapoints which are neighbors of the passed
        point.
    """
    point_distances = np.linalg.norm(data - point, axis=1)
    return np.argwhere(point_distances <= epsilon)[:, 0].tolist()


def dbscan(
    data: np.ndarray, epsilon: float, min_points: int = 4
) -> tuple[np.ndarray, int]:
    """Density-Based Spatial Clustering function.

    Generates an array of labels for each index of the passed data where:
        -1 - undefined point
        0 - noise
        k - point is part of cluster k

    Args:
        data: Numpy array containing the data with samples ordered along axis 0.
        point: The point for which to return the neighbours.
        epsilon: Max distance at which a datapoint is considered to be in a
            neighbor of the passed point.

    Returns:
        A numpy array of labels for each index of the passed data.
        An integer count of the number of created clusters.
    """
    labels = np.zeros(data.shape[0]) - 1
    k = 0
    for p in (points := tqdm(range(data.shape[0]))):
        if labels[p] > -1:
            continue  # if p is already labeled
        neighbors = _get_neighbors(data, data[p], epsilon)
        if len(neighbors) < min_points:
            labels[p] = 0  # if p is not a core point
            continue
        k += 1
        labels[p] = k
        n = -1
        while n < len(neighbors) and n < data.shape[0]:
            if labels[neighbors[n]] > -1:
                # includes the case where neighbors[n] == p
                n += 1
                continue
            labels[neighbors[n]] = k  # neighbors[n] is a border point of k
            new_neighbors = _get_neighbors(data, data[neighbors[n]], epsilon)
            if len(new_neighbors) >= min_points:
                neighbors += new_neighbors  # if neighbors[n] is a core point of k
            n += 1
        points.set_postfix(k=k, p=p, neighbors=len(neighbors))
    return labels, k


if __name__ == "__main__":
    """Clusters random generated demonstration data"""
    rng = np.random.default_rng()
    data = rng.random((500, 2))
    data[:150] += np.array([0.5, 1])
    labels, k = dbscan(data, 0.5, 20)

    for c in range(k):
        c += 1
        cluster = data[np.argwhere(labels == c)[:, 0]]
        plt.scatter(cluster[:, 0], cluster[:, 1])
    if 0 in labels:
        noise = data[np.argwhere(labels == 0)[0, :]]
        plt.scatter(noise[:, 0], noise[:, 1], marker="x", c="k")
    plt.show()
