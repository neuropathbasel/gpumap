# Author: Peter Eisenmann
#
# License: BSD 3 clause
from math import pow, ceil

from numba import cuda, int32
from numba.cuda.random import xoroshiro128p_uniform_float32, create_xoroshiro128p_states

import locale
import time

MAX_LOCAL = 16

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


@cuda.jit("f4(f4[:],f4[:])", device=True)
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
    for i in range(x.size):
        tmp = x[i] - y[i]
        result += tmp * tmp

    return result


@cuda.jit
def optimize_layout_cuda_simple(
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
    # unpack from various_type
    a = various_floats[0]
    b = various_floats[1]
    gamma = various_floats[2]
    initial_alpha = various_floats[3]

    move_other = write_embedding.shape[0] == read_embedding.shape[0]

    # edges handled by this thread
    thread_id = cuda.grid(1)
    n_threads = cuda.gridsize(1)
    thread_size = int(ceil(N_EDGES / n_threads))
    start_edge = thread_id * thread_size
    end_edge = min((thread_id + 1) * thread_size, N_EDGES)

    #iterate multiple times if local cache too small
    alpha = initial_alpha * (1.0 - (epoch / N_EPOCHS))

    ## normal algorithm
    for edge in range(start_edge, end_edge):
        if epoch_of_next_sample[edge] <= epoch:
            epoch_of_next_sample[edge] += epochs_per_sample[edge]

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

                for d in range(N_DIMS):
                    grad_d = clip(grad_coeff * (current[d] - other[d]))
                    cuda.atomic.add(current, d, grad_d * alpha)
                    if move_other:
                        cuda.atomic.add(other, d, -grad_d * alpha)

            # negative sampling
            for p in range(NEGATIVE_SAMPLE_RATE):
                # generate random number between 0 and N, not i or j
                k = (((((
                    int(xoroshiro128p_uniform_float32(rng_states, thread_id))
                    % (N_VERTICES - 2)) + i + 1)
                    % (N_VERTICES  - 1)) + j + 1)
                    % N_VERTICES)

                other = read_embedding[k]

                dist_squared = rdist(current, other)

                grad_coeff = 0.0
                grad_d = 4.0
                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1.0
                    )

                for d in range(N_DIMS):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                    cuda.atomic.add(current, d, grad_d * alpha)

@cuda.jit
def optimize_layout_cuda_random(
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
    # unpack from various_type
    a = various_floats[0]
    b = various_floats[1]
    gamma = various_floats[2]
    initial_alpha = various_floats[3]

    MOVE_OTHER = write_embedding.shape[0] == read_embedding.shape[0]

    # edges handled by this thread
    thread_id = cuda.grid(1)
    block_thread_id = cuda.threadIdx.x
    n_threads = cuda.gridsize(1)
    thread_size = int(ceil(N_EDGES / n_threads))
    current_edge = thread_id * thread_size
    edge_end = min((thread_id + 1) * thread_size, N_EDGES)

    #initiate local array
    prefiltered = cuda.local.array(shape=(MAX_LOCAL), dtype=int32)

    random = cuda.shared.array(shape=(1), dtype=int32)

    #iterate multiple times if local cache too small
    alpha = initial_alpha * (1.0 - (epoch / N_EPOCHS))
    while current_edge <= edge_end:
        #prefilter
        count = 0
        while current_edge <= edge_end and count < MAX_LOCAL:
            sample_edge = int(epoch_of_next_sample[current_edge] <= epoch)
            epoch_of_next_sample[current_edge] += float(sample_edge) * epochs_per_sample[current_edge]
            #store in local array
            prefiltered[count] = current_edge
            count += sample_edge
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

                for d in range(N_DIMS):
                    grad_d = clip(grad_coeff * (current[d] - other[d]))
                    cuda.atomic.add(current, d, grad_d * alpha)
                    if MOVE_OTHER:
                        cuda.atomic.add(other, d, -grad_d * alpha)
            cuda.syncthreads()

            # negative sampling
            for p in range(NEGATIVE_SAMPLE_RATE):
                if block_thread_id == 0:
                    random[0]=int(xoroshiro128p_uniform_float32(rng_states, cuda.blockIdx.x))

                cuda.syncthreads()

                # generate random number between 0 and N
                k = (random[0] + block_thread_id) % N_VERTICES

                # don't use self as negative sample
                k = (k + int(k == i)) % N_VERTICES # += 1 if equal

                other = read_embedding[k]

                dist_squared = rdist(current, other)

                grad_coeff = 0.0
                grad_d = 4.0

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1.0
                    )

                for d in range(N_DIMS):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                    cuda.atomic.add(current, d, grad_d * alpha)

                cuda.syncthreads()

        cuda.syncthreads()


@cuda.jit
def optimize_layout_cuda(
    write_embedding,
    read_embedding,
    head, # locally ascending
    tail, # ascending
    epochs_per_sample,
    epoch_of_next_sample,
    rng_states,
):
    # edges handled by this thread
    thread_id = cuda.grid(1)
    n_threads = cuda.gridsize(1)
    thread_size = int(ceil(N_EDGES / n_threads))
    start_edge = thread_id * thread_size
    edge_end = min((thread_id + 1) * thread_size, N_EDGES)

    #initiate local array
    prefiltered = cuda.local.array(shape=(MAX_LOCAL), dtype=int32)

    for epoch in range(N_EPOCHS):
        current_edge = start_edge
        alpha = INITIAL_ALPHA * (1.0 - (epoch / N_EPOCHS))

        #iterate multiple times if local cache too small
        while current_edge < edge_end:
            #prefilter
            count = 0
            while current_edge < edge_end and count < MAX_LOCAL:
                sample_edge = int(epoch_of_next_sample[current_edge] <= epoch)
                epoch_of_next_sample[current_edge] += float(sample_edge) * epochs_per_sample[current_edge]
                #store in local array
                prefiltered[count] = current_edge
                count += sample_edge
                current_edge += 1

            #all threads in block found up to max_local edges to sample
            cuda.syncthreads()

            ## normal algorithm
            for e in range(count):
                edge = prefiltered[e]

                # load nodes for edge
                i = tail[edge]
                j = head[edge]
                current = write_embedding[i]
                other = read_embedding[j]

                dist_squared = 0.0
                for d in range(N_DIMS):
                    tmp = current[d] - other[d]
                    dist_squared += tmp * tmp

                grad_coeff = 0.0

                if dist_squared > 0.0:
                    grad_coeff = -2.0 * A * B * pow(dist_squared, B - 1.0)
                    grad_coeff /= A * pow(dist_squared, B) + 1.0

                    for d in range(N_DIMS):
                        grad_d = clip(grad_coeff * (current[d] - other[d]))

                        cuda.atomic.add(current, d, grad_d * alpha)
                        if MOVE_OTHER:
                            cuda.atomic.add(other, d, -grad_d * alpha)
                cuda.syncthreads()

                # negative sampling
                for p in range(NEGATIVE_SAMPLE_RATE):
                    # generate random number between 0 and N, not i or j
                    k = (((((
                        int(xoroshiro128p_uniform_float32(rng_states, thread_id))
                        % (N_VERTICES - 2)) + i + 1)
                        % (N_VERTICES  - 1)) + j + 1)
                        % N_VERTICES)

                    other = read_embedding[k]

                    dist_squared = 0.0
                    for d in range(N_DIMS):
                        tmp = current[d] - other[d]
                        dist_squared += tmp * tmp

                    grad_coeff = 0.0
                    grad_d = 4.0
                    if dist_squared > 0.0:
                        grad_coeff = 2.0 * GAMMA * B
                        grad_coeff /= (0.001 + dist_squared) * (
                            A * pow(dist_squared, B) + 1.0
                        )

                    for d in range(N_DIMS):
                        if grad_coeff > 0.0:
                            grad_d = clip(grad_coeff * (current[d] - other[d]))
                        cuda.atomic.add(current, d, grad_d * alpha)

            cuda.syncthreads()
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

    # sanitize negative sample rate
    if verbose and negative_sample_rate != int(negative_sample_rate):
        print("rounding negative sample rate to ", int(negative_sample_rate))

    # copy arrays to device
    start = time.time()
    d_head_embedding = cuda.to_device(head_embedding)

    if MOVE_OTHER:
        d_tail_embedding = cuda.to_device(tail_embedding)
    else:
        d_tail_embedding = d_head_embedding

    d_head = cuda.to_device(head)
    d_tail = cuda.to_device(tail)

    d_epochs_per_sample = cuda.to_device(epochs_per_sample)
    d_epoch_of_next_sample = cuda.to_device(epochs_per_sample)

    d_rng_states = create_xoroshiro128p_states(n_threads,seed=rng_state[0])

    if verbose:
        end = time.time()
        print("Copying took {0} seconds".format(end-start))

    # run on gpu
    optimize_layout_cuda[n_blocks, threads_per_block](
        d_head_embedding,
        d_tail_embedding,
        d_head,
        d_tail,
        d_epochs_per_sample,
        d_epoch_of_next_sample,
        d_rng_states,
    )

    # copy result back from device
    head_embedding = d_head_embedding.copy_to_host()

    return head_embedding

