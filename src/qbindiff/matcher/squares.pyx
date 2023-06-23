"""
Copyright 2023 Quarkslab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

cimport numpy as cnp
from cython.parallel cimport prange
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.pair cimport pair

from qbindiff.utils.openmp_helpers import _openmp_effective_n_threads

cnp.import_array()

ctypedef cnp.npy_intp intp
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

    cdef intp begin, end, i, j, nodeA, nodeB, nodeC, nodeD

    cdef int rows = sparse_sim_matrix.shape[0]
    cdef int cols = sparse_sim_matrix.shape[1]

    cdef int[::1] indptr = sparse_sim_matrix.indptr
    cdef int[::1] indices = sparse_sim_matrix.indices

    cdef vector[vector[intp]] primary_children = vector[vector[intp]](rows)
    cdef vector[vector[intp]] secondary_children = vector[vector[intp]](cols)

    cdef unordered_set[int] sparse_sim_matrix_keys

    cdef vector[vector[iquadruple]] partial_squares = vector[vector[iquadruple]](rows)

    cdef int num_threads = _openmp_effective_n_threads()

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
                        partial_squares[i].push_back(
                            pair[ipair, ipair](
                                pair[intp, intp](nodeA, nodeB),
                                pair[intp, intp](nodeC, nodeD)
                            )
                        )

    # Merge the partials into a single list
    squares = []
    for i in range(rows):
        for p in partial_squares[i]:
            squares.append((p.first.first, p.first.second, p.second.first, p.second.second))

    return squares
