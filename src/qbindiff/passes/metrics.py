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

import numpy as np
import sklearn.metrics
from scipy.spatial import distance
from scipy.sparse import issparse, csr_matrix

from qbindiff.passes.fast_metrics import sparse_canberra


def canberra_distances(X, Y):
    """
    Compute the canberra distances between the vectors in X and Y.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        An array where each row is a sample and each column is a feature.

    Y : array-like of shape (n_samples_Y, n_features)
        An array where each row is a sample and each column is a feature.

    Returns
    -------
    D : ndarray of shape (n_samples_X, n_samples_Y)
        D contains the pairwise canberra distances.

    Notes
    -----
    When X and/or Y are CSR sparse matrices and they are not already
    in canonical format, this function modifies them in-place to
    make them canonical.
    """

    X, Y = sklearn.metrics.pairwise.check_pairwise_arrays(X, Y)

    if issparse(X) or issparse(Y):
        X = csr_matrix(X, copy=False)
        Y = csr_matrix(Y, copy=False)
        X.sum_duplicates()  # this also sorts indices in-place
        Y.sum_duplicates()
        D = np.zeros((X.shape[0], Y.shape[0]))
        sparse_canberra(X.data, X.indices, X.indptr, Y.data, Y.indices, Y.indptr, D)
        return D

    return distance.cdist(X, Y, "canberra")


CUSTOM_DISTANCES = {
    "canberra": canberra_distances,
}


def pairwise_distances(X, Y, metric="euclidean", *, n_jobs=None, **kwargs):
    """
    Compute the distance matrix from a vector array X and Y.
    The returned matrix is the pairwise distance between the arrays from both X and Y.

    In addition to the scikit-learn metrics, the following ones also work with
    sparse matrices: 'canberra'

    The backend implementation of the metrics rely on scikit-learn, refer to the manual
    of sklearn for more information:

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

    WARNING: if the metric is a callable then it must compute the distance between
    two matrices, not between two vectors. This is done so that the metric can optimize
    the calculations with parallelism.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features). The first feature matrix.

    Y : ndarray of shape (n_samples_Y, n_features), The second feature matrix.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a feature
        array. If metric is a string, it must be one of the supported metrics by
        scikit-learn.
        Alternatively, if metric is a callable function, it is called on the two input
        feature matrix (or a submatrix if n_jobs > 1). The callable should take two
        matrices as input and return a the resulting distance matrix.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This works by breaking down
        the pairwise matrix into n_jobs even slices and computing them in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    **kwargs : optional keyword parameters
        Any further parameters are passed directly to the scikit-learn implementation
        of pairwise_distances if a sklearn metric is used, otherwise they are passed
        to the callable metric specified.

    Returns
    -------
    D : ndarray of shape (n_samples_X, n_samples_Y)
        A distance matrix D such that D_{i, j} is the distance between the ith array
        from X and the jth array from Y.
    """

    if callable(metric):
        return sklearn.metrics.pairwise._parallel_pairwise(
            X, Y, metric, n_jobs, **kwargs
        )
    elif metric in CUSTOM_DISTANCES:
        return sklearn.metrics.pairwise._parallel_pairwise(
            X, Y, CUSTOM_DISTANCES[metric], n_jobs, **kwargs
        )
    else:
        return sklearn.metrics.pairwise.pairwise_distances(
            X, Y, metric, n_jobs=n_jobs, **kwargs
        )
