# Author: Peter Eisenmann
#
# License: BSD 3 clause
from __future__ import print_function
from warnings import warn

from math import pow

from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32

import locale

locale.setlocale(locale.LC_NUMERIC, "C")

@cuda.jit(device=True)
def rdist(x, y):
    """Reduced Euclidean distance.

    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)

    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    for i in range(x.shape[0]):
        tmp = (x[i] - y[i])
        result += tmp * tmp

    return result


@cuda.jit
def optimize_layout_cuda(
    write_embedding,
    read_embedding,
    head,
    tail,
    epoch,
    epochs_per_sample,
    epoch_of_next_sample,
    a,
    b,
    rng_states,
    gamma,
    alpha,
    negative_sample_rate,
):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).

    Parameters
    ----------
    write_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.

    read_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the write_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.

    head: array of shape (n_1_simplices)
        Indices to define ranges of nodes in adjacencies.

    tail: array of shape (n_1_simplices)
        All connected nodes to the node giving by the indexing in adjacency_indices

    epoch: int
        The number of the current training epoch.

    epochs_per_samples: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.

    epoch_of_next_sample: array of shape (n_1_simplices)
        A float value indicating the next epoch that a 1-simplex is to be sampled.

    a: float
        Parameter of differentiable approximation of right adjoint functor

    b: float
        Parameter of differentiable approximation of right adjoint functor

    rng_states: array of shape (
    int64, shape (3,)
        The internal state of the rng

    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.

    alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.

    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """
    thread_id = cuda.grid(1)
    n_threads = cuda.gridsize(1)
    n_dims = write_embedding.shape[1]
    n_vertices = read_embedding.shape[0]
    n_edges = head.shape[0]

    # edges handled by this thread
    thread_size = ceil(n_edges / n_threads)
    edge_start = thread_id * thread_size
    edge_end = min((thread_id + 1) * thread_size, n_edges)

    # for n in range(n_epochs): handled by caller
    for edge in range(edge_start, edge_end):
        if epoch_of_next_sample[edge] <= epoch:
            # load nodes for edge
            i = tail[edge]
            current = write_embedding[i]
            j = head[edge]
            other = read_embedding[j]
            
            dist_squared = rdist(current, other)

            grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
            grad_coeff /= a * pow(dist_squared, b) + 1.0
            
            if dist_squared <= 0.0:
                grad_coeff = 0.0

            for d in range(n_dims):
                grad_d = max(min(grad_coeff * (current[d] - other[d]),4.0),-4.0)
                current[d] += grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            for p in range(negative_sample_rate):
                k = int(xoroshiro128p_uniform_float32(rng_states, thread_id)) % n_vertices

                other = read_embedding[k]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1.0
                    )
                elif k==j:
                    continue
                else:
                    grad_coeff = 0.0

                for d in range(n_dims):
                    if grad_coeff > 0.0:
                        grad_d = max(min(grad_coeff * (current[d] - other[d]),4.0),-4.0)
                    else:
                        grad_d = 4.0
                    current[d] += grad_d * alpha

