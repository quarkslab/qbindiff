"""
BSD 3-Clause License

Copyright (c) 2007-2021 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

cimport numpy as cnp
from cython cimport floating
from cython.parallel cimport prange
from libc.math cimport fabs

from qbindiff.utils.openmp_helpers import _openmp_effective_n_threads

cnp.import_array()


def sparse_canberra(floating[::1] X_data, int[:] X_indices, int[:] X_indptr,
                    floating[::1] Y_data, int[:] Y_indices, int[:] Y_indptr,
                    double[:, ::1] D):
    """Pairwise canberra distances for CSR matrices"""
    cdef cnp.npy_intp px, py, i, j, ix, iy
    cdef double d = 0.0

    cdef int m = D.shape[0]
    cdef int n = D.shape[1]

    cdef int X_indptr_end = 0
    cdef int Y_indptr_end = 0

    cdef int num_threads = _openmp_effective_n_threads()

    # We scan the matrices row by row.
    # Given row px in X and row py in Y, we find the positions (i and j
    # respectively), in .indices where the indices for the two rows start.
    # If the indices (ix and iy) are the same, the corresponding data values
    # are processed and the cursors i and j are advanced.
    # If not, the lowest index is considered. Its associated data value is
    # processed and its cursor is advanced.
    # We proceed like this until one of the cursors hits the end for its row.
    # Then we process all remaining data values in the other row.

    # Below the avoidance of inplace operators is intentional.
    # When prange is used, the inplace operator has a special meaning, i.e. it
    # signals a "reduction"

    for px in prange(m, nogil=True, num_threads=num_threads):
        X_indptr_end = X_indptr[px + 1]
        for py in range(n):
            Y_indptr_end = Y_indptr[py + 1]
            i = X_indptr[px]
            j = Y_indptr[py]
            d = 0.0
            while i < X_indptr_end and j < Y_indptr_end:
                ix = X_indices[i]
                iy = Y_indices[j]

                if ix == iy:
                    d = d + fabs(X_data[i] - Y_data[j]) / (fabs(X_data[i]) + fabs(Y_data[j]))
                    i = i + 1
                    j = j + 1
                elif ix < iy:
                    d = d + 1
                    i = i + 1
                else:
                    d = d + 1
                    j = j + 1

            if i == X_indptr_end:
                d = d + Y_indptr_end - j
            else:
                d = d + X_indptr_end - i

            D[px, py] = d
