# Author: Peter Eisenmann
#
# License: BSD 3 clause
import numpy as np

from math import sqrt, ceil

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
    n_samples = X.shape[0]
    n_dims = X.shape[1]

    # Simple implementation. Basically also brute force, but does not need as
    # much memory as bruteForceKnn for unknown reasons. Performs maybe half a
    # second slower for small data sets

    # assure that data is contiguous, otherwise FAISS can not process it
    X = np.ascontiguousarray(X.astype(np.float32))

    resource = faiss.StandardGpuResources()

    index = faiss.GpuIndexFlatL2(resource, n_dims)
    index.train(X)
    index.add(X)

    #knn_dists, knn_indices = index.search(X, n_neighbors)

    # query in batches above certain limit
    # TODO determine limit, requires to known device memory size, possible input
    if n_samples < 500000 :
        knn_dists, knn_indices = index.search(X, n_neighbors)
    else:
        # query 5 times, since FAISS reserves approximately 18% of memory as
        # temporary memory and thus querying 5 times always allows each query
        # to fit in memory
        n_queries = 5
        slice_size = ceil(n_samples / n_queries)
        knn_dists = np.zeros((n_samples, n_neighbors), dtype=np.float32)
        knn_indices = np.zeros((n_samples, n_neighbors), dtype=np.int64)
        for i in range(n_queries):
            start = i * slice_size
            end = min(start + slice_size, n_samples)
            knn_dists[start:end], knn_indices[start:end] = index.search(X[start:end], n_neighbors)

    return knn_indices, knn_dists, []

    # bruteForceKnn method. Performs as fast as simple version, but easily runs
    # into memory errors.
#    n_samples = X.shape[0]
#    n_dims = X.shape[1]

#    resource = faiss.StandardGpuResources()
#    metric = faiss.METRIC_L2

#    X_flat = np.ascontiguousarray(X).astype(np.float32)
#    knn_dists = np.empty((n_samples, n_neighbors),dtype=np.float32)
#    knn_indices = np.empty((n_samples, n_neighbors),dtype=np.int64)

#    X_ptr = faiss.swig_ptr(X_flat)
#    knn_indices_ptr = faiss.swig_ptr(knn_indices)
#    knn_dists_ptr = faiss.swig_ptr(knn_dists)

#    faiss.bruteForceKnn(resource, metric, X_ptr, n_samples, X_ptr, n_samples, n_dims, n_neighbors, knn_dists_ptr, knn_indices_ptr)

#    # FAISS outputs each point as closest to itself, but the rest of the code
#    # does not seem to mind, so leaving them in place. Otherwise the first of
#    # each indices row and distances rows would need to be removed.

#    return knn_indices, knn_dists, []


    # Code as FAISS is used by t-SNE-CUDA. It performs slower then both methods
    # above and is therefore not used.
    # Requires: from math import sqrt
#    X = np.ascontiguousarray(X.astype(np.float32))

#    resource = faiss.StandardGpuResources()

#    kNumCells = int(sqrt(float(n_samples)))
#    kNumCellsToProbe = 20

#    # Construct the GPU configuration object
#    config = faiss.GpuIndexIVFFlatConfig()
#    config.device = 0
#    config.indicesOptions = faiss.INDICES_32_BIT
#    config.flatConfig.useFloat16 = False
#    config.useFloat16IVFStorage = False

#    index = faiss.GpuIndexIVFFlat(resource, n_dims, kNumCells, faiss.METRIC_L2, config)
#    index.setNumProbes(kNumCellsToProbe)
#    index.train(X) #n_samples, points)
#    index.add(X) #n_samples, points)

#    # Perform the KNN query
#    knn_dists, knn_indices = index.search(X, n_neighbors)

#    return knn_indices, knn_dists, []


