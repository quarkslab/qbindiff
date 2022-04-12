import logging
from typing import Generator

import numpy as np

# Import for types
from qbindiff.types import Positive, Ratio
from qbindiff.types import RawMapping, Vector, SparseMatrix


class BeliefMWM:

    """
    Computes the optimal solution to the **Maxmimum Weight Matching problem**.
    """

    def __init__(self, similarity: SparseMatrix, epsilon: Positive = 0.5):
        self._init_indices(similarity)
        self._init_messages()

        self._objective = []
        self._maxscore = 0.0
        self._epsilon = self._dtype(epsilon)
        self._epsilonref = self._epsilon.copy()

    def compute(self, maxiter: int = 1000):
        for niter in range(1, maxiter + 1):
            self._update_messages()
            self._round_messages()
            self._update_epsilon()
            yield niter
            if self._converged():
                logging.info("[+] Converged after %i iterations" % niter)
                return
        logging.info("[+] Did not converged after %i iterations" % maxiter)

    @property
    def current_mapping(self) -> RawMapping:
        rows = np.searchsorted(self._rowmap, self._mates.nonzero()[0], side="right") - 1
        cols = self._colidx[self._mates]
        mask = np.intersect1d(
            np.unique(rows, return_index=True)[1], np.unique(cols, return_index=True)[1]
        )
        return rows[mask], cols[mask]

    @property
    def current_score(self) -> float:
        return self._weigths[self._mates].sum()

    def _init_indices(self, similarity: SparseMatrix):
        self._weigths = similarity.data.copy()
        self._shape = similarity.shape
        self._dtype = similarity.dtype.type
        self._colidx = similarity.indices
        self._rowmap = similarity.indptr
        self._colmap = np.zeros(similarity.shape[1] + 1, dtype=np.uint32)
        self._colmap[1:] = np.bincount(
            similarity.indices, minlength=similarity.shape[1]
        ).cumsum(dtype=np.uint32)
        self._tocol = np.argsort(similarity.indices, kind="mergesort").astype(np.uint32)
        self._torow = np.zeros(similarity.nnz, dtype=np.uint32)
        self._torow[self._tocol] = np.arange(similarity.nnz, dtype=np.uint32)

    def _init_messages(self):
        self._x = self._weigths.copy()
        self._y = self._weigths.copy()
        self._messages = np.zeros_like(self._weigths)
        self._mates = np.zeros_like(self._weigths, dtype=bool)

    def _update_messages(self):
        if self._shape[0] < self._shape[1]:
            self._update_messages2()
        else:
            self._update_messages1()

    def _update_messages1(self):
        self._x[:] = self._weigths
        self._x += self._other_colmax(self._y)  # axis=0
        self._messages[:] = self._x
        self._y[:] = self._weigths
        self._y += self._other_rowmax(self._x)  # axis=1
        self._messages += self._x

    def _update_messages2(self):
        self._x[:] = self._weigths
        self._x += self._other_rowmax(self._y)  # axis=1
        self._messages[:] = self._x
        self._y[:] = self._weigths
        self._y += self._other_colmax(self._x)  # axis=0
        self._messages += self._x

    def _round_messages(self):
        self._mates[:] = self._messages > 0
        self._objective.append(self.current_score)

    @property
    def _rowslice(self) -> Generator[Vector, None, None]:
        return map(slice, self._rowmap[:-1], self._rowmap[1:])

    @property
    def _colslice(self) -> Generator[Vector, None, None]:
        return map(slice, self._colmap[:-1], self._colmap[1:])

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

    def _othermax(self, vector: Vector):
        """
        Compute the maximum value for all elements except (for the maxmimum value)
        $$x_i = max_{j!=i}{x_j}$$
        """
        if len(vector) > 1:
            arg2, arg1 = np.argpartition(vector, -2)[-2:]
            max2, max1 = np.maximum(0.0, vector[[arg2, arg1]], dtype=self._dtype)
            vector[:] = -max1 - self._epsilon
            vector[arg1] = -max2
        else:
            vector[:] = 0.0

    def _update_epsilon(self):
        if len(self._objective) < 10:
            return
        current_score = self._objective[-1] / max(self._mates.sum(), 1)
        if self._maxscore >= current_score:
            self._epsilon *= 1.2
        else:
            self.best_mapping = self.current_mapping
            self._best_messages = self._messages.copy()
            self._maxscore = current_score
            self._epsilon = self._epsilonref

    def _converged(self, window: int = 60, pattern: int = 15) -> bool:
        """
        Decide whether or not the algorithm have converged

        :param m: minimum size of the pattern to match
        :param w: latest score of the w last function matching

        :return: True or False if the algorithm have converged
        :rtype: bool
        """
        objective = self._objective[: -window + 1 : -1]
        if len(objective) > pattern:
            score = objective[0]
            if score in objective[pattern:]:
                pivot = objective[pattern:].index(score) + pattern
                if objective[:pattern] == objective[pivot : pivot + pattern]:
                    return True
        return False


class BeliefQAP(BeliefMWM):

    """
    Computes an approximate solution to the **Quadratic Assignment problem**.
    """

    def __init__(
        self,
        similarity: SparseMatrix,
        squares: SparseMatrix,
        tradeoff: Ratio = 0.5,
        epsilon: Positive = 0.5,
    ):
        super(BeliefQAP, self).__init__(similarity, epsilon=epsilon)
        if tradeoff == 1:
            logging.warning("[+] meaningless tradeoff for NAQP")
            squares -= squares
        else:
            self._weigths *= 2 * tradeoff / (1 - tradeoff)
        self._init_squares(squares)

    @property
    def current_score(self) -> float:
        objective = super(BeliefQAP, self).current_score
        objective += self.numsquares * 2
        return objective

    @property
    def numsquares(self) -> int:
        return self._z[self._mates][:, self._mates].nnz / 2

    def _init_squares(self, squares: SparseMatrix):
        self._z = squares.astype(self._dtype)
        self._zmax = self._z.data.copy()
        self._zrownnz = np.diff(squares.indptr)
        self._ztocol = np.argsort(squares.indices, kind="mergesort")
        np.clip(self._z.data, 0, self._zmax, out=self._z.data)

    def _update_messages1(self):
        self._messages[:] = self._weigths
        self._messages += self._z.sum(0).getA1()
        self._x[:] = self._messages
        self._x += self._other_colmax(self._y)  # axis=0
        self._y[:] = self._messages
        self._messages[:] = self._x
        self._y += self._other_rowmax(self._x)  # axis=1
        self._messages += self._x

    def _update_messages2(self):
        self._messages[:] = self._weigths
        self._messages += self._z.sum(0).getA1()
        self._x[:] = self._messages
        self._x += self._other_rowmax(self._y)  # axis=1
        self._y[:] = self._messages
        self._messages[:] = self._x
        self._y += self._other_colmax(self._x)  # axis=0
        self._messages += self._x

    def _round_messages(self):
        super(BeliefQAP, self)._round_messages()
        self._clip_z()

    def _clip_z(self):
        # self._messages -= self._epsilon * (1 - self._mates)
        np.take(self._z.data, self._ztocol, out=self._z.data)
        np.negative(self._z.data, out=self._z.data)
        self._z.data += np.repeat(self._messages, self._zrownnz)
        self._z.data += self._zmax
        np.clip(self._z.data, 0, self._zmax, out=self._z.data)
