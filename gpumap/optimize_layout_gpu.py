# Author: Peter Eisenmann
#
# License: BSD 3 clause
from math import pow, ceil

from numba import cuda, int64
from numba.cuda.random import xoroshiro128p_uniform_float32, create_xoroshiro128p_states

import locale

import time

locale.setlocale(locale.LC_NUMERIC, "C")

MAX_LOCAL = 16

@cuda.jit("f4(f4)", device=True)
def clip(val):
    """Standard clamping of a value into a fixed range (in this case -4.0 to
    4.0)

    Parameters
    ----------
    val: float
        The value to be clamped.

    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    return max(min(val, 4.0), -4.0)


@cuda.jit("f4(f4[:],f4[:])", device=True,)
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
        tmp = x[i] - y[i]
        result += tmp * tmp

    return result


@cuda.jit
def optimize_layout_cuda_prefilter(
    write_embedding,
    read_embedding,
    head, # locally ascending
    tail, # ascending
    epoch,
    epochs_per_sample,
    epoch_of_next_sample,
    various_floats,
    rng_states,
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

    various_floats: array of shape (4)
        An array containg the float values: a, b, gamma, initial_alpha (in that
        order).

    rng_states: array of shape (n_threads)
        The internal states of the rng for each thread.

    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """
    # unpack from various_type
    a = various_floats[0]
    b = various_floats[1]
    gamma = various_floats[2]
    initial_alpha = various_floats[3]
    n_epochs = various_floats[4]
    negative_sample_rate = various_floats[5]

    alpha = initial_alpha * (1.0 - (epoch / n_epochs))
    move_other = write_embedding.shape[0] == read_embedding.shape[0]

    # sizes
    n_dims = write_embedding.shape[1]
    n_vertices = read_embedding.shape[0]
    n_edges = head.shape[0]

    # edges handled by this thread
    thread_id = cuda.grid(1)
    n_threads = cuda.gridsize(1)
    thread_size = int(ceil(n_edges / n_threads))
    current_edge = thread_id * thread_size
    edge_end = min((thread_id + 1) * thread_size, n_edges)

    #initiate local array
    prefiltered = cuda.local.array(shape=(MAX_LOCAL), dtype=int64)

    #iterate multiple times if local cache too small
    while current_edge <= edge_end:
        #prefilter
        count = 0
        while current_edge <= edge_end and count < MAX_LOCAL:
            if epoch_of_next_sample[current_edge] <= epoch:
                epoch_of_next_sample[current_edge] += epochs_per_sample[current_edge]
                #store in local array
                prefiltered[count] = current_edge
                count += 1
            current_edge += 1

        #all threads in block found up to max_local edges to sample
        cuda.syncthreads()

        ## normal algorithm
        for e in range(count):
            edge = prefiltered[e]

            # load nodes for edge
            i = tail[edge]
            current = write_embedding[i]
            j = head[edge]
            other = read_embedding[j]

            dist_squared = rdist(current, other)

            grad_coeff = 0.0

            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0

                for d in range(n_dims):
                    grad_d = clip(grad_coeff * (current[d] - other[d]))
                    current[d] += grad_d * alpha
                    if move_other:
                        other[d] += -grad_d * alpha
#                    cuda.atomic.add(current, d, grad_d * alpha)
#                    if move_other:
#                        cuda.atomic.add(other, d, -grad_d * alpha)
            cuda.syncthreads()

            for p in range(int(negative_sample_rate)):
                # generate random number between 0 and N, not i or j
                k = (((((
                    int(xoroshiro128p_uniform_float32(rng_states, thread_id))
                    % (n_vertices - 2)) + i + 1)
                    % (n_vertices  - 1)) + j + 1)
                    % n_vertices)

                other = read_embedding[k]

                dist_squared = rdist(current, other)

                grad_coeff = 0.0
                grad_d = 4.0
                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1.0
                    )

                for d in range(n_dims):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                    current[d] += grad_d * alpha
#                    cuda.atomic.add(current, d, grad_d * alpha)

        cuda.syncthreads()

def optimize_layout_gpu(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma,
    initial_alpha,
    negative_sample_rate,
    verbose,
):
    # define thread and block amounts
    #TODO test multiple values
    n_threads = 1024
    threads_per_block = 32
    n_blocks = n_threads // threads_per_block #use dividable values

    # sanitize negative sample rate
    if verbose and negative_sample_rate != int(negative_sample_rate):
        print("rounding negative sample rate to ", int(negative_sample_rate))
    negative_sample_rate = int(negative_sample_rate)
    if negative_sample_rate > threads_per_block:
        warn("negative_sample_rate is set too high, lowering to " + str(threads_per_block))
        negative_sample_rate = threads_per_block

#    n_epochs = n_epochs * 2 #0 #TODO rm

    move_other = head_embedding.shape[0] == tail_embedding.shape[0]

    # copy arrays to device
    start = time.time()
    d_head_embedding = cuda.to_device(head_embedding)
    if move_other:
        d_tail_embedding = cuda.to_device(tail_embedding)
    else:
        d_tail_embedding = d_head_embedding

    d_head = cuda.to_device(head)
    d_tail = cuda.to_device(tail)

    d_epochs_per_sample = cuda.to_device(epochs_per_sample)
    d_epoch_of_next_sample = cuda.to_device(epochs_per_sample)

    d_various_floats = cuda.to_device((
        a,
        b,
        gamma,
        initial_alpha
        n_epochs,
        negative_sample_rate
    ))
    d_rng_states = create_xoroshiro128p_states(n_blocks,seed=rng_state[0])

    if verbose:
        end = time.time()
        print("Copying took {0} seconds".format(end-start))
    
    # run on gpu
    for n in range(n_epochs): #comment for _true
        optimize_layout_cuda_prefilter[n_blocks, threads_per_block](
            d_head_embedding,
            d_tail_embedding,
            d_head,
            d_tail,
            n,
            d_epochs_per_sample,
            d_epoch_of_next_sample,
            d_various_floats,
            d_rng_states,
        )

    # copy result back from device
    head_embedding = d_head_embedding.copy_to_host()

    return head_embedding

