# Author: Leland McInnes <leland.mcinnes@gmail.com>
# Contributer: Peter Eisenmann <p3732@gmx.de>
#
# License: BSD 3 clause
from math import ceil, exp, fabs, floor

import numpy as np
from numba import cuda

@cuda.jit(device=True)
def calculate_rho(distances, local_connectivity, smooth_k_tolerance):
    rho = 0.0

    index = int(floor(local_connectivity))
    max_non_zero_dist = 0.0
    distances_mean = 0.0
    first_non_zero_dist = 0.0
    non_zero_dist_at_index = 0.0
    non_zero_dist_at_index_minus_one = 0.0
    n_positive_distances = 0
    
    for dist in distances:
        if dist > 0.0:
            max_non_zero_dist = max(dist, max_non_zero_dist)
            distances_mean += dist
            if n_positive_distances == index:
                non_zero_dist_at_index = dist
            if n_positive_distances == index - 1:
                non_zero_dist_at_index_minus_one = dist
            n_positive_distances += 1
    if n_positive_distances > 0:
        distances_mean /= float(n_positive_distances)
    
    if n_positive_distances >= local_connectivity:
        interpolation = local_connectivity - index
        if index > 0:
            rho = non_zero_dist_at_index_minus_one
            if interpolation > smooth_k_tolerance:
                rho += interpolation * (non_zero_dist_at_index - non_zero_dist_at_index_minus_one)
        else:
            rho = interpolation * first_non_zero_dist
    else:
        rho = max_non_zero_dist

    return rho, distances_mean,


@cuda.jit(device=True)
def binary_search_knn_dist(
    distances, n_iter, rho_i, target, smooth_k_tolerance, 
):
    lo = 0.0
    hi = np.inf
    mid = 1.0

    for n in range(n_iter):
        psum = 0.0
        for j in range(1, len(distances)):
            d = distances[j] - rho_i
            if d > 0:
                psum += exp(-(d / mid))
            else:
                psum += 1.0

        if fabs(psum - target) < smooth_k_tolerance:
            break

        if psum > target:
            hi = mid
            mid = (lo + hi) / 2.0
        else:
            lo = mid
            if hi == np.inf:
                mid *= 2
            else:
                mid = (lo + hi) / 2.0
    return mid


@cuda.jit
def smooth_knn_dist_cuda(
        distances, rho, result, target, n_iter, local_connectivity,
        distances_mean, smooth_k_tolerance, min_k_dist_scale
    ):
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In esscence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.

    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each samples. Each row should be a
        sorted list of distances to a given samples nearest neighbors.

    k: float
        The number of nearest neighbors to approximate for.

    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.

    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    thread_id = cuda.grid(1)
    n_threads = cuda.gridsize(1)
    n_distances = distances.shape[0]
    thread_size = int(ceil(n_distances / n_threads))
    start_distance = thread_id * thread_size
    end_distance = min((thread_id + 1) * thread_size, n_distances)

    for i in range(start_distance, end_distance):
        ith_distances = distances[i]
        rho_i, ith_distances_mean = calculate_rho(
            ith_distances, local_connectivity, smooth_k_tolerance
        )
        
        result_i = binary_search_knn_dist(
            ith_distances,
            n_iter,
            rho_i,
             target,
              smooth_k_tolerance, 
        )

        if rho_i > 0.0:
            result[i] = max(result_i, min_k_dist_scale * ith_distances_mean)
        else:
            result[i] = max(result_i, min_k_dist_scale * distances_mean)
        rho[i] = rho_i

def smooth_knn_dist_gpu(
    distances, k, n_iter, local_connectivity, bandwidth, smooth_k_tolerance,
    min_k_dist_scale
):
    rho = np.zeros(distances.shape[0])
    result = np.zeros(distances.shape[0])

    target = np.log2(k) * bandwidth
    distances_mean = np.mean(distances)

    # define thread and block amounts
    #TODO test multiple values
    n_threads = 4096
    threads_per_block = 32
    n_blocks = n_threads // threads_per_block #use dividable values

    # copy arrays
    d_distances = cuda.to_device(distances)
    d_rho = cuda.to_device(rho)
    d_result = cuda.to_device(result)

    smooth_knn_dist_cuda[n_blocks, threads_per_block](
        d_distances, d_rho, d_result, target, n_iter, local_connectivity,
        distances_mean, smooth_k_tolerance, min_k_dist_scale
    )

    # copy results back from device
    result = d_result.copy_to_host()
    rho = d_rho.copy_to_host()

    return result, rho

