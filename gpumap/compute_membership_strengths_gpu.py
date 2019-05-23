# Author: Peter Eisenmann <p3732@gmx.de>
#
# License: BSD 3 clause
from math import ceil, exp

import numpy as np
import cupy as cp
from numba import cuda, float32

N_SAMPLES = 0
N_NEIGHBORS = 0

@cuda.jit()
def compute_membership_strengths_cuda(
    knn_indices, knn_dists, sigmas, rhos, rows, cols, vals
):
    thread_id = cuda.grid(1)
    n_threads = cuda.gridsize(1)
    thread_size = int(ceil(N_SAMPLES / n_threads))
    start_pos = thread_id * thread_size
    end_pos = min((thread_id + 1) * thread_size, N_SAMPLES)

    for i in range(start_pos, end_pos):
        for j in range(N_NEIGHBORS):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0:
                val = 1.0
            else:
                val = exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * N_NEIGHBORS + j] = i
            cols[i * N_NEIGHBORS + j] = knn_indices[i, j]
            vals[i * N_NEIGHBORS + j] = val


def compute_membership_strengths_gpu(
    d_knn_indices, d_knn_dists, d_sigmas, d_rhos
):
    global N_SAMPLES, N_NEIGHBORS
    N_SAMPLES   = d_knn_indices.shape[0]
    N_NEIGHBORS = d_knn_indices.shape[1]

    # create new device arrays
    array_size = N_SAMPLES * N_NEIGHBORS
    d_rows = cp.zeros((array_size), dtype=cp.int64)
    d_cols = cp.zeros((array_size), dtype=cp.int64)
    d_vals = cp.zeros((array_size), dtype=cp.float64)

    # define thread and block amounts
    n_threads = 8192
    threads_per_block = 128
    n_blocks = n_threads // threads_per_block #use dividable values

    compute_membership_strengths_cuda[n_blocks, threads_per_block](
        d_knn_indices, d_knn_dists, d_sigmas, d_rhos, d_rows, d_cols, d_vals
    )

    return d_rows, d_cols, d_vals

