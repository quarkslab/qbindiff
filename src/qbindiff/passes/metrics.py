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

from qbindiff.passes.fast_metrics import sparse_canberra, sparse_strong_jaccard


def _validate_vector(u, dtype=None):
    u = np.asarray(u, dtype=dtype)
    if u.ndim == 1:
        return u
    raise ValueError("Input vector should be 1-D.")


def _validate_weights(w, dtype=np.double):
    w = _validate_vector(w, dtype=dtype)
    if np.any(w < 0):
        raise ValueError("Input weights should be all non-negative")
    return w


def canberra_distances(X, Y, w=None):
    """
    Compute the canberra distances between the vectors in X and Y using the optional
    array of weights w.

    :param X: array-like of shape (n_samples_X, n_features)
              An array where each row is a sample and each column is a feature.

    :param Y: array-like of shape (n_samples_Y, n_features)
              An array where each row is a sample and each column is a feature.

    :param w: array-like of size n_features.
              The weights for each value in ``X`` and ``V``.
              Default is None, which gives each value a weight of 1.0

    :return D: ndarray of shape (n_samples_X, n_samples_Y)
               D contains the pairwise canberra distances.

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

        if w is not None:
            w = _validate_weights(w)
            if w.size != X.shape[1]:
                ValueError("Weights size mismatch")
        sparse_canberra(X.data, X.indices, X.indptr, Y.data, Y.indices, Y.indptr, D, w)
        return D

    if w is None:
        return distance.cdist(X, Y, "canberra")
    ValueError("Cannot assign weights with non-sparse matrices")


def jaccard_strong(X, Y, w=None):
    r"""
    Compute a variation of the jaccard distances between the vectors in X and Y using
    the optional array of weights w.

    The distance function between two vector ``u`` and ``v`` is the following:

    .. math::

        \sum_{i}\frac{f(u_i, v_i)}{ | \{ i | u_i \neq 0 \lor v_i \neq 0 \} | }

    where the function ``f`` is defined like this:

    .. math::

        f(x, y) = \begin{cases} 0 & \text{if } x = 0 \lor y = 0 \\ 1 - \frac{|x - y|}{|x| + |y|} & \text{otherwise.} \end{cases}

    If the optional weights are specified the formula becomes:

    .. math::

        \sum_{i}\frac{w_i * f(u_i, v_i)}{ | \{ i | u_i \neq 0 \lor v_i \neq 0 \} | }

    :param X: array-like of shape (n_samples_X, n_features)
              An array where each row is a sample and each column is a feature.

    :param Y: array-like of shape (n_samples_Y, n_features)
              An array where each row is a sample and each column is a feature.

    :param w: array-like of size n_features. The weights for each value in ``X`` and ``V``.
              Default is None, which gives each value a weight of 1.0

    :return D: ndarray of shape (n_samples_X, n_samples_Y)
               D contains the pairwise strong jaccard distances.

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

        if w is not None:
            w = _validate_weights(w)
            if w.size != X.shape[1]:
                ValueError("Weights size mismatch")
        sparse_strong_jaccard(
            X.data, X.indices, X.indptr, Y.data, Y.indices, Y.indptr, D, w
        )
        return D

    if w is None:
        return sparse_strong_jaccard(
            X.data, X.indices, X.indptr, Y.data, Y.indices, Y.indptr, D, None
        )
    ValueError("Cannot assign weights with non-sparse matrices")


CUSTOM_DISTANCES = {
    "canberra": canberra_distances,
    "jaccard-strong": jaccard_strong,
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

    **WARNING**: if the metric is a callable then it must compute the distance between
    two matrices, not between two vectors. This is done so that the metric can optimize
    the calculations with parallelism.

    :param X: ndarray of shape (n_samples_X, n_features). The first feature matrix.

    :param Y: ndarray of shape (n_samples_Y, n_features), The second feature matrix.

    :param metric: str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a feature
        array. If metric is a string, it must be one of the supported metrics by
        scikit-learn.
        Alternatively, if metric is a callable function, it is called on the two input
        feature matrix (or a submatrix if n_jobs > 1). The callable should take two
        matrices as input and return a the resulting distance matrix.

    :param n_jobs: int, default=None
        The number of jobs to use for the computation. This works by breaking down
        the pairwise matrix into n_jobs even slices and computing them in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    :param \*\*kwargs: optional keyword parameters
        Any further parameters are passed directly to the scikit-learn implementation
        of pairwise_distances if a sklearn metric is used, otherwise they are passed
        to the callable metric specified.

    :return D: ndarray of shape (n_samples_X, n_samples_Y)
        A distance matrix D such that D_{i, j} is the distance between the ith array
        from X and the jth array from Y.
    """

    if callable(metric):
        return sklearn.metrics.pairwise._parallel_pairwise(
            X, Y, metric, n_jobs, **kwargs
        )
    elif metric in CUSTOM_DISTANCES:
        # All the custom distances are guaranteed to make use of parallelism
        return CUSTOM_DISTANCES[metric](X, Y, **kwargs)
    else:
        return sklearn.metrics.pairwise.pairwise_distances(
            X, Y, metric, n_jobs=n_jobs, **kwargs
        )
