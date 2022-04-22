import logging

import numpy as np

# Import for types
from typing import Generator
from qbindiff.types import Positive, Ratio, RawMapping, Vector, SparseMatrix


class BeliefMWM:

    """
    Computes the optimal solution to the **Maxmimum Weight Matching problem**.
    """

    def __init__(self, sim_matrix: SparseMatrix, epsilon: Positive = 0.5):
        # The weights sparse matrix
        self.weights = sim_matrix.copy()
        self._shape = sim_matrix.shape
        self._dtype = sim_matrix.dtype.type

        self._init_messages()

        self.scores = []
        self.max_avg_score = 0.0
        self.epsilon = self._dtype(epsilon)
        self._epsilonref = self.epsilon.copy()

    def _init_messages(self):
        # Messages from node to factor targeting the node in the first graph. m(X[ii`] -> f[i])
        self.msg_n2f = self.weights.copy()
        # Messages from node to factor targeting the node in the second graph. m(X[ii`] -> g[i`])
        self.msg_n2g = self.weights.copy()
        # Messages from factor to node targeting the node in the first graph. m(f[i] -> X[ii`])
        self.msg_f2n = self.weights.copy()
        # Messages from factor to node targeting the node in the second graph. m(g[i`] -> X[ii`])
        self.msg_g2n = self.weights.copy()
        # Messages to the node, also known as max-marginal probability of the node. P(X[ii`])
        self.marginals = self.weights.copy()

        # The matching matrix between the two graphs. It is a mask that has to be applied
        # to self.weights.data
        self.matches_mask = np.zeros_like(self.weights.data, dtype=bool)

    def compute(self, maxiter: int = 1000):
        for niter in range(1, maxiter + 1):
            self.update_messages()
            self.round_messages()
            self.update_epsilon()
            yield niter
            if self.converged():
                logging.info("[+] Converged after %i iterations" % niter)
                return
        logging.info("[+] Did not converged after %i iterations" % maxiter)

    def update_messages(self):
        """Update the messages considering if it's better to start from the first graph or the second"""
        if self._shape[0] <= self._shape[1]:
            self.update_messages_primary()
        else:
            self.update_messages_secondary()

    def update_messages_primary(self):
        """Update messages starting from the first graph"""

        # Update messages from node to f
        self.msg_n2f.data[:] = self.weights.data
        self.update_factor_g_messages()
        self.msg_n2f.data += self.msg_g2n.data

        self.marginals.data[:] = self.msg_n2f.data

        # Update messages fron node to g
        self.msg_n2g.data[:] = self.weights.data
        self.update_factor_f_messages()
        self.msg_n2g.data += self.msg_f2n.data

        self.marginals.data += self.msg_f2n.data

    def update_messages_secondary(self):
        """Update messages starting from the second graph"""

        # Update messages from node to g
        self.msg_n2g.data[:] = self.weights.data
        self.update_factor_f_messages()
        self.msg_n2g.data += self.msg_f2n.data

        self.marginals.data[:] = self.msg_n2g.data

        # Update messages from node to f
        self.msg_n2f.data[:] = self.weights.data
        self.update_factor_g_messages()
        self.msg_n2f.data += self.msg_g2n.data

        self.marginals.data += self.msg_g2n.data

    def update_factor_msg(self, messages):
        """Update the messages from factor to node. It is done in-place."""
        if len(messages) > 1:
            arg2, arg1 = np.argpartition(messages, -2)[-2:]
            max2, max1 = np.maximum(0, messages[[arg2, arg1]], dtype=self._dtype)
            messages[:] = -max1 - self.epsilon
            messages[arg1] = -max2
        else:
            messages[:] = self._dtype(0)

    def update_factor_g_messages(self):
        """Update all the messages from factor g to node"""
        for k in range(self._shape[1]):
            col = self.msg_n2g[:, k]
            self.update_factor_msg(col.data)
            self.msg_g2n[:, k] = col

    def update_factor_f_messages(self):
        """Update all the messages from factor f to node"""
        for k in range(self._shape[0]):
            row = self.msg_n2f[k]
            self.update_factor_msg(row.data)
            self.msg_f2n[k] = row

    def round_messages(self):
        self.matches_mask[:] = self.marginals.data > 0
        self.scores.append(self.current_score)

    def update_epsilon(self) -> None:
        if len(self.scores) < 10:
            return
        avg_score = self.scores[-1] / max(self.matches_mask.sum(), 1)
        if self.max_avg_score >= avg_score:
            self.epsilon *= 1.2
        else:
            self.best_mapping = self.current_mapping
            self.best_marginals = self.marginals.copy()
            self.max_avg_score = avg_score
            self.epsilon = self._epsilonref

    def has_converged(self, window: int = 60, pattern_size: int = 15) -> bool:
        """
        Decide whether or not the algorithm has converged.
        The algorithm has converged if we can find the same pattern at least once by looking
        at the last `window` elements of the scores. The pattern is a list composed of the
        last `pattern_size` elements of the scores.

        :param window: Number of the latest scores to consider when searching for the pattern
        :param pattern_size: Size of the pattern

        :return: True or False if the algorithm have converged
        :rtype: bool
        """
        scores = self.scores[: -window - 1 : -1]
        if len(scores) < 2 * pattern_size:
            return False

        pattern = scores[:pattern_size]
        for i in range(pattern_size, window - pattern_size + 1):
            if pattern == scores[i : i + pattern_size]:
                return True
        return False

    @property
    def current_mapping(self) -> RawMapping:
        rows = (
            np.searchsorted(
                self.weights.indptr, self.matches_mask.nonzero()[0], side="right"
            )
            - 1
        )
        cols = self.weights.indices[self.matches_mask]
        mask = np.intersect1d(
            np.unique(rows, return_index=True)[1], np.unique(cols, return_index=True)[1]
        )
        return rows[mask], cols[mask]

    @property
    def current_score(self) -> float:
        return self.weights.data[self.matches_mask].sum()


class BeliefQAP(BeliefMWM):

    """
    Computes an approximate solution to the **Quadratic Assignment problem**.
    """

    def __init__(
        self,
        sim_matrix: SparseMatrix,
        squares: SparseMatrix,
        tradeoff: Ratio = 0.5,
        epsilon: Positive = 0.5,
    ):
        super(BeliefQAP, self).__init__(sim_matrix, epsilon)
        if tradeoff == 1:
            logging.warning("[+] meaningless tradeoff for NAQP")
            squares -= squares
        else:
            self.weights.data *= 2 * tradeoff / (1 - tradeoff)
        self._init_squares(squares)

    @property
    def current_score(self) -> float:
        score = super(BeliefQAP, self).current_score
        score += self.numsquares * 2
        return score

    @property
    def numsquares(self) -> int:
        squares = self.msg_h2n[self.matches_mask][:, self.matches_mask]
        return (squares.sum() + squares.diagonal().sum()) / 2

    def _init_squares(self, squares: SparseMatrix):
        # Messages from square factor to node. m(h[ii`jj`] -> X[ii`])
        self.msg_h2n = squares.astype(self._dtype)

        # The additional weight matrix addressing the squares weights. W[ii`jj`]
        self.weights_squares = self.msg_h2n.data.copy()

        # Number of squares (ii`, jj`) for each edge ii`
        self.squares_per_edge = np.diff(squares.indptr)

    def update_messages_primary(self):
        """Update messages starting from the first graph"""

        partial = self.weights.data.copy()
        partial += self.msg_h2n.sum(1).getA1()

        # Update messages from node to f
        self.msg_n2f.data[:] = partial
        self.update_factor_g_messages()
        self.msg_n2f.data += self.msg_g2n.data

        self.marginals.data[:] = self.msg_n2f.data

        # Update messages fron node to g
        self.msg_n2g.data[:] = partial
        self.update_factor_f_messages()
        self.msg_n2g.data += self.msg_f2n.data

        self.marginals.data += self.msg_f2n.data

    def update_messages_secondary(self):
        """Update messages starting from the second graph"""

        partial = self.weights.data.copy()
        partial += self.msg_h2n.sum(1).getA1()

        # Update messages from node to g
        self.msg_n2g.data[:] = partial
        self.update_factor_f_messages()
        self.msg_n2g.data += self.msg_f2n.data

        self.marginals.data[:] = self.msg_n2g.data

        # Update messages from node to f
        self.msg_n2f.data[:] = partial
        self.update_factor_g_messages()
        self.msg_n2f.data += self.msg_g2n.data

        self.marginals.data += self.msg_g2n.data

    def round_messages(self):
        super(BeliefQAP, self).round_messages()
        self.update_square_factor_messages()

    def update_square_factor_messages(self):
        # partial is the message from node to square factor m(X[ii`] -> h[ii`jj`])
        partial = self.msg_h2n.copy()
        np.negative(partial.data, out=partial.data)
        partial.data += np.repeat(self.marginals.data, self.squares_per_edge)

        # transpose
        partial = partial.T.tocsr()
        positive_partial = np.clip(partial.data, 0, max(0, partial.data.max()))

        tmp = self.weights_squares + partial.data
        np.clip(tmp, 0, tmp.max(), out=tmp)

        self.msg_h2n.data[:] = tmp - positive_partial
