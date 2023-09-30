# Copyright 2023 Quarkslab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for fast squares matrix generation
"""

cimport numpy as cnp
cimport openmp
import numpy as np
from scipy.sparse import csr_matrix
from cython.parallel cimport prange
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair

from qbindiff.utils.openmp_helpers import _openmp_effective_n_threads

cnp.import_array()

ctypedef cnp.npy_intp intp
ctypedef cnp.npy_uint32 uint32
ctypedef cnp.npy_uint64 uint64
ctypedef pair[intp, intp] ipair
ctypedef pair[ipair, ipair] iquadruple


def find_squares(
    cnp.npy_bool[:, ::1] primary_adj_matrix,
    cnp.npy_bool[:, ::1] secondary_adj_matrix,
    sparse_sim_matrix
):
    """
    Find all the squares given two graphs described by their adjacency matrix and a
    sparse similarity matrix (CSR format).
    For a definition of a square see Matcher._compute_squares_matrix() doc
    
    Returns a list of tuples (nodeA, nodeB, nodeC, nodeD), each representing a square.
    the elements of the tuple are the indexes of the nodes.
    """

    cdef intp begin, end, i, j
    cdef uint64 r, c, nodeA, nodeB, nodeC, nodeD
    cdef uint32 edge_idx1, edge_idx2

    cdef int rows = sparse_sim_matrix.shape[0]
    cdef int cols = sparse_sim_matrix.shape[1]
    cdef int size = sparse_sim_matrix.nnz

    cdef int[::1] indptr = sparse_sim_matrix.indptr
    cdef int[::1] indices = sparse_sim_matrix.indices

    cdef vector[vector[intp]] primary_children = vector[vector[intp]](rows)
    cdef vector[vector[intp]] secondary_children = vector[vector[intp]](cols)

    cdef unordered_map[uint64, uint32] edge_map

    cdef unordered_set[int] sparse_sim_matrix_keys

    # Initialize the rows and columns of the square sparse matrix
    cdef vector[uint32] squares_rows
    cdef vector[uint32] squares_cols

    cdef int num_threads = _openmp_effective_n_threads()

    # Mutex for synchronization
    cdef openmp.omp_lock_t lock
    openmp.omp_init_lock(&lock)

    # Populate the children per node
    for i in range(rows):
        for j in range(rows):
            if primary_adj_matrix[i,j]:
                primary_children[i].push_back(j)
    for i in range(cols):
        for j in range(cols):
            if secondary_adj_matrix[i,j]:
                secondary_children[i].push_back(j)

    # similarity matrix fast access
    for i in range(rows):
        begin = indptr[i]
        end = indptr[i+1]
        for j in range(begin, end):
            sparse_sim_matrix_keys.insert(i*cols + indices[j])

    # Map for fast lookup of node pairs (used as indexes in the square matrix)
    # This means giving each similarity edge a unique number.
    #  (nodeA, nodeB) -> idx
    bipartite = sparse_sim_matrix.tocoo().astype(np.uint32)
    bipartite.data[:] = np.arange(0, size, dtype=np.uint32)
    for i in range(size):
        r = bipartite.row[i]
        c = bipartite.col[i]
        edge_map[r * cols + c] = bipartite.data[i]

    # Find the squares
    for i in prange(rows, nogil=True, num_threads=num_threads):
        begin = indptr[i]
        end = indptr[i+1]
        nodeA = i
        if primary_children[nodeA].empty():
            continue

        for j in range(begin, end):
            nodeB = indices[j]
            if secondary_children[nodeB].empty():
                continue

            for nodeC in secondary_children[nodeB]:
                for nodeD in primary_children[nodeA]:
                    if sparse_sim_matrix_keys.find(cols*nodeD + nodeC) != sparse_sim_matrix_keys.end():
                        edge_idx1 = edge_map[nodeA * cols + nodeB]
                        edge_idx2 = edge_map[nodeD * cols + nodeC]

                        openmp.omp_set_lock(&lock)

                        # Add the suqares (A, B, C, D) and (C, D, A, B)
                        squares_rows.push_back(edge_idx1)
                        squares_rows.push_back(edge_idx2)
                        squares_cols.push_back(edge_idx2)
                        squares_cols.push_back(edge_idx1)

                        openmp.omp_unset_lock(&lock)

    # Sparse square matrix data
    data = np.ones(squares_rows.size(), dtype=np.uint8)

    # Convert the C++ vectors in numpy arrays.
    # This is important to avoid huge spikes in memory usage when calling numpy.asarray
    # Note that it would be even better to avoid using C++ vectors entirely but we don't know
    # beforehand the size of the numpy arrays (numpy arrays cannot be resized cheaply at runtime)
    np_squares_rows = np.zeros(squares_rows.size(), dtype=np.uint32)
    np_squares_cols = np.zeros(squares_cols.size(), dtype=np.uint32)
    for i in range(squares_rows.size()):
        np_squares_rows[i] = squares_rows[i]
        np_squares_cols[i] = squares_cols[i]

    # Build csr matrix
    squares_matrix = csr_matrix((data, (np_squares_rows, np_squares_cols)), shape=(size, size), dtype=np.uint8)

    # Sometimes a square is counted twice
    # ex: (nodeA, nodeB, nodeC, nodeD) == (nodeC, nodeD, nodeA, nodeB)
    # Set the data to ones to count all the squares only once
    squares_matrix.data[:] = 1

    return squares_matrix
