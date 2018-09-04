## coding: utf-8

import logging
import numpy as np
from scipy.sparse import csr_matrix, diags
from itertools import chain

from qbindiff.types import Generator, R, CallGraph, InputMatrix, Vector, BeliefMatching, Iterator


class BeliefMatrixError(Exception):
    pass


class BeliefMWM(object):
    """
    Compute the **Maxmimum Weight Matching** of the matrix of weights (similarity).
    Returns the *real* maximum assignement.
    """
    def __init__(self, weights: csr_matrix):
        weights = self._checkmatrix(weights)
        self.weights = weights.data
        self.mates = np.zeros_like(self.weights, dtype=bool)
        self.objective = []
        self._init_indices(weights)
        self._init_messages()

    def compute_matching(self, maxiter: int=100) -> Generator[int, None, None]:
        niter = 0
        for _ in range(maxiter):
            self._update_messages()
            niter += 1
            yield niter
            if self._converged():
                for _ in range(self._converged_iter):
                    self._update_messages()
                    niter += 1
                    yield niter
                yield maxiter
                logging.debug("Converged after %d iterations" % niter)
                return
        logging.debug("Did not converge after %d iterations" % niter)

    @property
    def matching(self) -> BeliefMatching:
        rows = np.logical_or.reduceat(self.mates, self._rowmap[:-1]).nonzero()[0]
        cols = self._colidx[self.mates]
        weights = self.weights[self.mates]
        return zip(rows, cols, weights)

    def _init_indices(self, weights: csr_matrix) -> None:
        self.dims = weights.shape
        self._colidx = weights.indices
        self._rowmap = weights.indptr
        self._colmap = np.hstack((0, np.bincount(self._colidx).cumsum(dtype=np.int32)))
        self._tocol = self._colidx.argsort(kind="mergesort").astype(np.int32)
        self._torow = np.zeros_like(self._tocol)
        self._torow[self._tocol] = np.arange(weights.size, dtype=np.int32)
        self._rownnz = np.diff(weights.indptr)

    def _init_messages(self) -> None:
        self.x = self.weights.astype(np.float32)
        self.y = self.weights.astype(np.float32)
        self.messages = np.zeros_like(self.weights, np.float32)

    def _update_messages(self) -> None:
        self.x[:] = self.weights + self._other_rowmax(self.y)
        self.messages[:] = self.x
        self.y[:] = self.weights + self._other_colmax(self.x)
        self.messages += self.x
        self._round_messages()

    def _round_messages(self) -> None:
        self.mates[:]= self.messages >= 0
        matchmask = np.add.reduceat(self.mates, self._rowmap[:-1]) == 1
        self.mates[:] &= np.repeat(matchmask, self._rownnz)
        self.objective.append(self._objective())

    @property
    def _rowslice(self):
        return map(slice, self._rowmap[:-1], self._rowmap[1:])

    @property
    def _colslice(self):
        return map(slice, self._colmap[:-1], self._colmap[1:])
    """
    def _rowsum(self, vector, values):
        for row, value in zip(self._rowslice, values):
            vector[row] += value

    def _rowand(self, vector, values):
        for row, value in zip(self._rowslice, values):
            vector[row] &= value
    """
    def _other_rowmax(self, vector: Vector) -> Vector:
        for row in self._rowslice:
            self._othermax(vector[row])
        return vector

    def _other_colmax(self, vector: Vector) -> Vector:
        vector.take(self._tocol, out=vector)
        for col in self._colslice:
            self._othermax(vector[col])
        vector.take(self._torow, out=vector)
        return vector

    @staticmethod
    def _othermax(vector: Vector) -> None:
        """
        Compute the maximum value for all elements except (for the maxmimum value)
        $$x_i = max_{j!=i}{x_j}$$
        """
        if len(vector) > 1:
            arg2, arg1 = np.argpartition(vector, -2)[-2:]
            max2, max1 = - np.maximum(0, vector[[arg2, arg1]])
            vector[:] = max1
            vector[arg1] = max2
        else:
            vector[:] = 0

    def _objective(self) -> float:
        return self.weights[self.mates].sum()

    def _converged(self, m: int=5, w: int=50) -> bool:
        """
        Decide whether or not the algorithm have converged

        :param m: minimum size of the pattern to match
        :param w: latest score of the w last function matching

        :return: True or False if the algorithm have converged
        :rtype: bool
        """
        def _converged_(obj, idx):
            return obj[-2*idx:-idx] == obj[-idx:]
        patterns = self.objective[-w:-m]
        actual = self.objective[-1]
        if actual in patterns:
            pivot = patterns[::-1].index(actual) + m
            if _converged_(self.objective, pivot):
                self._converged_iter = np.argmax(self.objective[-pivot:]) + 1
                return True
        return False

    @staticmethod
    def _checkmatrix(matrix: InputMatrix) -> csr_matrix:
        """
        Normalize the weight values into something homogenous
        """
        try:
            matrix = csr_matrix(matrix)
        except Exception:
            raise BeliefMatrixError("Unknown matrix type: %s" % str(type(matrix)))
        if not (matrix.getnnz(0).all() and matrix.getnnz(1).all()):
            raise BeliefMatrixError("Incomplete bipartite, (isolated nodes)")
        return matrix


class BeliefNAQP(BeliefMWM):
    """
    Compute an approximate solution to **Network Alignement Quadratic Problem**.
    """
    def __init__(self, weights: InputMatrix, edges1: CallGraph, edges2: CallGraph, tradeoff: float=0.5, active_beta=False):
        super(BeliefNAQP, self).__init__(weights)
        self._init_squares(weights, edges1, edges2)
        self._init_beta(tradeoff, active_beta)

    def _init_squares(self, weights: InputMatrix, edges1: CallGraph, edges2: CallGraph) -> None:
        self.z = self.compute_squares(weights, edges1, edges2)
        self.mz = np.zeros_like(self.weights, np.float32)
        self._zrownnz = np.diff(self.z.indptr)
        self._ztocol = np.argsort(self.z.indices, kind="mergesort")

    def _init_beta(self, tradeoff, active_beta):
        self.active_beta = active_beta
        if tradeoff == 0:
            self.weights = np.zeros_like(self.weights)
            tradeoff = .5
        tradeoff = 1 / tradeoff - 1
        if active_beta:
            self.beta = np.full_like(weights.data, tradeoff)
        else:
            self.beta = tradeoff

    def _update_messages(self) -> None:
        self.mz[:] = self.weights + self.z.sum(0).getA1()
        self.x[:] = self.mz + self._other_rowmax(self.y)
        self.messages[:] = self.x
        self.y[:] = self.mz + self._other_colmax(self.x)
        self.messages += self.x
        self._round_messages()

        self._clip_beta()

    def _clip_beta(self):
        if self.active_beta:
            beta = np.repeat(self.beta, self._zrownnz)
        else:
            beta = self.beta
        self.messages += beta
        #self.z.data[self._ztocol] = - self.z.data
        np.take(self.z.data, self._ztocol, out=self.z.data)
        np.negative(self.z.data, out=self.z.data)
        self.z.data += np.repeat(self.messages, self._zrownnz)
        np.clip(self.z.data, 0, beta, out=self.z.data)

    def _round_messages(self) -> None:
        if self.active_beta:
            messages = self.messages >= 0
            matchmask = np.add.reduceat(messages, self._rowmap[:-1]) == 1
            messages &= np.repeat(matchmask, self._rownnz)
            self.beta += self.mates & messages
            self.mates = messages
            self.objective.append(self._objective())
        else:
            super(BeliefNAQP, self)._round_messages()

    def _objective(self) -> float:
        objective = super(BeliefNAQP, self)._objective()
        objective += self.numsquares
        return objective

    @property
    def numsquares(self) -> int:
        return self.z[self.mates][:, self.mates].nnz / 2

    @staticmethod
    def compute_squares(weights: csr_matrix, edges1: CallGraph, edges2: CallGraph) -> csr_matrix:
        size = weights.nnz
        bipartite = weights.astype(bool)
        bipartite.data = np.arange(1, len(bipartite.data)+1, dtype=np.int32)
        indices = bipartite.indices
        indptr = bipartite.indptr
        bipartite = bipartite.toarray()
        edgenum = np.array(list(map(len, edges2)))
        rowslice = map(indices.__getitem__, map(slice, indptr[:-1], indptr[1:]))
        idxx, idxy = [], []
        for i, j in enumerate(rowslice):
            edgesi = edges1[i]
            edgesj = list(chain(*map(edges2.__getitem__,  j)))
            contacts = bipartite[np.ix_(edgesi, edgesj)]
            contacted = contacts.nonzero()
            #idx = indptr[i] + np.searchsorted(edgenum[j].cumsum(), contacted[1], side="right")
            idx = indptr[i] + np.repeat(np.arange(len(j)), edgenum[j])[contacted[1]]
            idy = contacts[contacted] - 1
            idxx.extend(idx.tolist())
            idxy.extend(idy.tolist())
        boolean = np.ones_like(idxx, bool)
        S = csr_matrix((boolean, (idxx, idxy)), shape=(size, size), dtype=np.float32)
        S += S.T
        return S

