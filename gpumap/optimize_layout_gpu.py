# Author: Peter Eisenmann
#
# License: BSD 3 clause
from math import pow, ceil, floor

import cupy as cp
import cupy as cp

from numba import cuda, float32, int32
from numba.cuda.random import xoroshiro128p_uniform_float32, create_xoroshiro128p_states

import locale
import time

MAX_LOCAL = 32
EPSILON = 0.000000000001

# used as constants during compilation, allowing loop unrolling
N_VERTICES = 0
N_DIMS = 0
N_EDGES = 0
N_EPOCHS = 0
NEGATIVE_SAMPLE_RATE = 0

A = 0.0
B = 0.0
GAMMA = 0.0
INITIAL_ALPHA = 0.0

MOVE_OTHER = True

@cuda.jit("f4(f4)", device=True, inline=True)
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

@cuda.jit
def optimize_layout_cuda(
    write_embedding,
    read_embedding,
    head, # locally ascending
    tail, # ascending
    epoch,
    epochs_per_sample,
    epoch_of_next_sample,
    rng_states,
):
    # edges handled by this thread
    thread_id = cuda.grid(1)
    n_threads = cuda.gridsize(1)
    thread_size = int(floor(N_EDGES / n_threads))
    offset = N_EDGES % n_threads
    current_edge = thread_id * thread_size + min(offset, thread_id)
    end_edge = (thread_id + 1) * thread_size + min(offset, thread_id + 1)

    #initiate local array
    prefiltered = cuda.local.array(shape=(MAX_LOCAL), dtype=int32)
    cached = cuda.local.array(shape=(MAX_LOCAL), dtype=float32)

    alpha = INITIAL_ALPHA * (1.0 - (epoch / float(N_EPOCHS)))

    # init cache
    start_node = tail[current_edge]
    end_node = tail[max(end_edge - 1, 0)]
    cached_i = start_node
    for d in range(N_DIMS):
        cached[d] = read_embedding[cached_i][d]

    #iterate multiple times if local cache too small
    while current_edge < end_edge:
        #prefilter
        count = 0
        while current_edge < end_edge and count < MAX_LOCAL:
            sample_edge = int(epoch_of_next_sample[current_edge] <= epoch)
            epoch_of_next_sample[current_edge] += float(sample_edge) * epochs_per_sample[current_edge]
            #store in local array
            prefiltered[count] = current_edge
            count += sample_edge
            current_edge += 1

        # All threads in block found up to MAX_LOCAL edges to sample, start
        # normal algorithm
        for e in range(count):
            edge = prefiltered[e]

            # load nodes for edge
            i = tail[edge]
            j = head[edge]
            if not cached_i == i:
                if cached_i == start_node:
                    for d in range(N_DIMS):
                        # might be written by other thread, use average
                        prev = write_embedding[cached_i][d]
                        write_embedding[cached_i][d] = 0.5 * prev + 0.5 * cached[d]
                else:
                    for d in range(N_DIMS):
                        write_embedding[cached_i][d] = cached[d]
                for d in range(N_DIMS):
                    cached[d] = read_embedding[i][d]
                cached_i = i

            other = read_embedding[j]

            dist_squared = 0.0
            for d in range(N_DIMS):
                tmp = cached[d] - other[d]
                dist_squared += tmp * tmp

            grad_coeff = 0.0

            if dist_squared > EPSILON:
                grad_coeff = -2.0 * A * B * pow(dist_squared, B - 1.0)
                grad_coeff /= A * pow(dist_squared, B) + 1.0

                for d in range(N_DIMS):
                    grad_d = clip(grad_coeff * (cached[d] - other[d]))
                    neg_grad_d = clip(grad_coeff * (other[d] - cached[d]))
                    cached[d] += grad_d * alpha
                    cached[d] -= neg_grad_d * alpha

            # negative sampling
            for p in range(NEGATIVE_SAMPLE_RATE):
                # generate random number between 0 and N, not i or j
                k = (((((
                    int(xoroshiro128p_uniform_float32(rng_states, thread_id))
                    % (N_VERTICES - 2)) + i + 1)
                    % (N_VERTICES - 1)) + j + 1)
                    % N_VERTICES)

                other = read_embedding[k]

                dist_squared = 0.0
                for d in range(N_DIMS):
                    tmp = cached[d] - other[d]
                    dist_squared += tmp * tmp

                grad_coeff = 0.0
                grad_d = 4.0
                if dist_squared > EPSILON:
                    grad_coeff = 2.0 * GAMMA * B
                    grad_coeff /= (0.001 + dist_squared) * (
                        A * pow(dist_squared, B) + 1.0
                    )

                for d in range(N_DIMS):
                    if grad_coeff > EPSILON:
                        grad_d = clip(grad_coeff * (cached[d] - other[d]))
                    cached[d] += grad_d * alpha
        cuda.syncthreads()
    if cached_i == end_node:
        for d in range(N_DIMS):
            # might be written by other thread, use average
            prev = write_embedding[cached_i][d]
            write_embedding[cached_i][d] = 0.5 * prev + 0.5 * cached[d]
    else:
        for d in range(N_DIMS):
            write_embedding[cached_i][d] = cached[d]


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
    rng_states,
    gamma,
    initial_alpha,
    negative_sample_rate,
    verbose,
):
    # define thread and block amounts
    #TODO test multiple values
    n_threads = 8192
    threads_per_block = 128
    n_blocks = n_threads // threads_per_block #use dividable values

    global N_DIMS, N_VERTICES, N_EDGES, N_EPOCHS, NEGATIVE_SAMPLE_RATE
    N_VERTICES = head_embedding.shape[0]
    N_DIMS = head_embedding.shape[1]
    N_EDGES = head.shape[0]
    N_EPOCHS = n_epochs
    NEGATIVE_SAMPLE_RATE = int(negative_sample_rate)

    global A, B, GAMMA, INITIAL_ALPHA
    A = a
    B = b
    GAMMA = gamma
    INITIAL_ALPHA = initial_alpha

    global MOVE_OTHER
    MOVE_OTHER = head_embedding.shape[0] == tail_embedding.shape[0]

    # create remaining arrays
    if MOVE_OTHER:
        tail_embedding = cp.copy(head_embedding)
    else:
        tail_embedding = cp.asarray(tail_embedding)
    epoch_of_next_sample = cp.copy(epochs_per_sample)

    # run on gpu
    write_embedding = head_embedding
    read_embedding = tail_embedding

    if verbose:
        print("move other is set to", MOVE_OTHER)

    for epoch in range(N_EPOCHS):
        if MOVE_OTHER:
            tmp = read_embedding
            read_embedding = write_embedding
            write_embedding = tmp

        optimize_layout_cuda[n_blocks, threads_per_block](
            write_embedding,
            read_embedding,
            head,
            tail,
            epoch,
            epochs_per_sample,
            epoch_of_next_sample,
            rng_states,
        )

        if verbose and epoch % int(N_EPOCHS / 10) == 0:
            print("\tcompleted ", epoch, " / ", N_EPOCHS, "epochs")
    if verbose:
        print("\tcompleted ", N_EPOCHS, " / ", N_EPOCHS, "epochs")

    return write_embedding.get()

