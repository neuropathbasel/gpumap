# Author: Peter Eisenmann
#
# License: BSD 3 clause
import numpy as np

import faiss

# Implementation based on
# https://github.com/erikbern/ann-benchmarks/blob/master/ann_benchmarks/algorithms/faiss_gpu.py
# which itself is based on
# https://github.com/facebookresearch/faiss/blob/master/benchs/bench_gpu_sift1m.py

def nearest_neighbors_gpu(X, n_neighbors):
    """Compute the ``n_neighbors`` nearest points for each data point in ``X``.
    This may be exact, but more likely is approximated via nearest neighbor
    search of the faiss library.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The input data to compute the k-neighbor graph of.

    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.

    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.
    """
    X = np.ascontiguousarray(X.astype(np.float32))
    n_samples = X.shape[0]
    n_dims = X.shape[1]
    resource = faiss.StandardGpuResources()
    
    index = faiss.GpuIndexFlatL2(resource, n_dims)
    index.train(X)
    index.add(X)
    
    knn_dists, knn_indices = index.search(X, n_neighbors)

    if np.any(knn_indices < 0):
        warn(
            "Failed to correctly find n_neighbors for some samples."
            "Results may be less than ideal. Try re-running with"
            "different parameters."
        )

    return knn_indices, knn_dists, []

