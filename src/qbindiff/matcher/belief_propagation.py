# Copyright 2023 Quarkslab

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Belief propagation framework

Contains the whole belief propagation implementation.
"""

import logging
import math
import numpy as np
from typing import Any
from collections.abc import Generator

# local imports
from qbindiff.types import Positive, Ratio, RawMapping, SparseMatrix


class BeliefMWM:
    """
    Computes the optimal solution to the **Maxmimum Weight Matching problem**.
    """

    def __init__(self, sim_matrix: SparseMatrix, epsilon: Positive = 0.5):
        """
        :param sim_matrix: similarity matrix (sparse numpy matrix)
        :param epsilon: perturbation for algorithm convergence
        """
        self.weights = sim_matrix.copy()  #: The weights sparse matrix
        self._shape = sim_matrix.shape
        self._dtype = sim_matrix.dtype.type

        self._init_messages()

        self.scores: list[float] = []  #: Scores list
        self.max_avg_score: float = 0.0  #: Current maximum average score
        self.best_mapping: RawMapping = None  #: Current best mapping
        self.best_marginals = None  #: Current associated marginals as a SparseMatrix
        self.epsilon = self._dtype(epsilon)  #: Current epsilon
        self._epsilonref = self.epsilon.copy()

    def _init_messages(self) -> None:
        """
        Initializes messages for the belief propagation phase
        """
        #: Messages from node to factor targeting the node in the first graph. m(X[ii`] -> f[i])
        self.msg_n2f = self.weights.copy()
        #: Messages from node to factor targeting the node in the second graph. m(X[ii`] -> g[i`])
        self.msg_n2g = self.weights.copy()
        #: Messages from factor to node targeting the node in the first graph. m(f[i] -> X[ii`])
        self.msg_f2n = self.weights.copy()
        #: Messages from factor to node targeting the node in the second graph. m(g[i`] -> X[ii`])
        self.msg_g2n = self.weights.copy()
        #: Messages to the node, also known as max-marginal probability of the node. P(X[ii`])
        self.marginals = self.weights.copy()

        # The matching matrix between the two graphs. It is a mask that has to be applied
        # to self.weights.data
        self.matches_mask = np.zeros_like(self.weights.data, dtype=bool)

    def compute(self, maxiter: int = 1000) -> Generator[int, Any, Any]:
        """
        Repeat the belief propagation round for a given number of iterations

        :param maxiter: Maximum number of iterations for the algorithm
        :return: generator that yield at each iteration
        """

        for niter in range(1, maxiter + 1):
            self._update_messages()
            self._round_messages()
            self._update_epsilon()
            yield niter
            if self._has_converged():
                logging.info(f"[+] Converged after {niter} iterations")
                return
        logging.info(f"[+] Did not converged after {maxiter} iterations")

    def _update_messages(self) -> None:
        """
        Update the messages considering if it's better to start from the first graph or the second
        """
        if self._shape[0] <= self._shape[1]:
            self._update_messages_primary()
        else:
            self._update_messages_secondary()

    def _update_messages_primary(self) -> None:
        """
        Update messages starting from the first graph
        """

        # Update messages from node to f
        self.msg_n2f.data[:] = self.weights.data
        self._update_factor_g_messages()
        self.msg_n2f.data += self.msg_g2n.data

        self.marginals.data[:] = self.msg_n2f.data

        # Update messages from node to g
        self.msg_n2g.data[:] = self.weights.data
        self._update_factor_f_messages()
        self.msg_n2g.data += self.msg_f2n.data

        self.marginals.data += self.msg_f2n.data

    def _update_messages_secondary(self) -> None:
        """
        Update messages starting from the second graph
        """

        # Update messages from node to g
        self.msg_n2g.data[:] = self.weights.data
        self._update_factor_f_messages()
        self.msg_n2g.data += self.msg_f2n.data

        self.marginals.data[:] = self.msg_n2g.data

        # Update messages from node to f
        self.msg_n2f.data[:] = self.weights.data
        self._update_factor_g_messages()
        self.msg_n2f.data += self.msg_g2n.data

        self.marginals.data += self.msg_g2n.data

    def _update_factor_msg(self, messages) -> None:
        """
        Update the messages from factor to node. It is done in-place.

        :param messages: messages to update
        """
        if len(messages) > 1:
            arg2, arg1 = np.argpartition(messages, -2)[-2:]
            max2, max1 = np.maximum(0, messages[[arg2, arg1]], dtype=self._dtype)
            messages[:] = -max1 - self.epsilon
            messages[arg1] = -max2
        else:
            messages[:] = self._dtype(0)

    def _update_factor_g_messages(self) -> None:
        """
        Update all the messages from factor g to node
        """

        # Use the csc (compressed sparse column) format for efficiency
        msg_n2g_csc = self.msg_n2g.tocsc()
        self.msg_g2n = self.msg_g2n.tocsc()

        for k in range(self._shape[1]):
            # All the messages share the same sparse matrix structure, i.e. they all
            # have the same indptr and the same indices arrays
            # This lets us perform some optimizations
            begin = msg_n2g_csc.indptr[k]
            end = msg_n2g_csc.indptr[k + 1]
            col = msg_n2g_csc.data[begin:end]
            self._update_factor_msg(col)
            self.msg_g2n.data[begin:end] = col

            # Non optimized version
            # ~ col = self.msg_n2g[:, k]
            # ~ self.update_factor_msg(col.data)
            # ~ self.msg_g2n[:, k] = col

        # Restore the csr (compressed sparse row) format
        self.msg_g2n = self.msg_g2n.tocsr()

    def _update_factor_f_messages(self) -> None:
        """
        Update all the messages from factor f to node
        """
        for k in range(self._shape[0]):
            # All the messages share the same sparse matrix structure, i.e. they all
            # have the same indptr and the same indices arrays
            # This lets us perform some optimizations
            begin = self.msg_n2f.indptr[k]
            end = self.msg_n2f.indptr[k + 1]
            row = self.msg_n2f.data[begin:end]
            self._update_factor_msg(row)
            self.msg_f2n.data[begin:end] = row

            # Non optimized version
            # ~ row = self.msg_n2f[k]
            # ~ self.update_factor_msg(row.data)
            # ~ self.msg_f2n[k] = row

    def _round_messages(self) -> None:
        """
        Rounding phase
        """
        self.matches_mask[:] = self.marginals.data > 0
        self.scores.append(self.current_score)

    def _update_epsilon(self) -> None:
        """
        Epsilon phase
        """
        avg_score = self.scores[-1] / max(self.matches_mask.sum(), 1)
        if self.max_avg_score < avg_score:
            self.best_mapping = self.current_mapping
            self.best_marginals = self.marginals.copy()
            self.max_avg_score = avg_score
            if len(self.scores) >= 10:
                self.epsilon = self._epsilonref
        elif len(self.scores) >= 10:
            self.epsilon *= 1.2

    def _has_converged(self, window: int = 60, pattern_size: int = 15) -> bool:
        """
        Decide whether the algorithm has converged.
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
        """
        Current mapping
        """

        rows = (
            np.searchsorted(self.weights.indptr, self.matches_mask.nonzero()[0], side="right") - 1
        )
        cols = self.weights.indices[self.matches_mask]
        mask = np.intersect1d(
            np.unique(rows, return_index=True)[1], np.unique(cols, return_index=True)[1]
        )
        return rows[mask], cols[mask]

    @property
    def current_score(self) -> float:
        """Current score"""
        return self.weights.data[self.matches_mask].sum()

    @property
    def current_marginals(self) -> SparseMatrix:
        """
        Current marginals in a sparse matrix
        """

        curr_marginals = self.marginals.copy()
        # The output of np.power might results in +inf, hence we need to clip those
        # values. Here it is clipped to [0, 1e6] since 1e6/(1e6+1) ~ 0.999999
        # Since those values are real probabilities it means that all the
        # values > 99.9999% are the same.
        curr_marginals.data[:] = [
            x / (1 + x) for x in np.clip(np.power(math.e, curr_marginals.data), 0, 1e6)
        ]
        return curr_marginals


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
        """
        :param sim_matrix: similarity matrix (sparse numpy matrix)
        :param squares: square matrix
        :param tradeoff: trade-off value (close to 0 similarity, close to 1 squares (callgraph))
        :param epsilon: perturbation value for convergence
        """
        super(BeliefQAP, self).__init__(sim_matrix, epsilon)
        if tradeoff == 1:
            logging.warning("[+] meaningless tradeoff for NAQP")
            squares -= squares
        else:
            self.weights.data *= 2 * tradeoff / (1 - tradeoff)
        self._init_squares(squares)

    @property
    def current_score(self) -> float:
        """Current score of the solution"""
        score = super(BeliefQAP, self).current_score
        score += self.numsquares * 2
        return score

    @property
    def numsquares(self) -> int:
        """Number of squares"""
        squares = self.msg_h2n[self.matches_mask][:, self.matches_mask]
        return (squares.sum() + squares.diagonal().sum()) / 2

    def _init_squares(self, squares: SparseMatrix) -> None:
        """
        Initializes the square matrix

        :param squares: square matrix
        """

        #: Messages from square factor to node. m(h[ii`jj`] -> X[ii`])
        self.msg_h2n = squares.astype(self._dtype)

        #: The additional weight matrix addressing the squares weights. W[ii`jj`]
        self.weights_squares = self.msg_h2n.data.copy()

        #: Number of squares (ii`, jj`) for each edge ii`
        self.squares_per_edge = np.diff(squares.indptr)

    def _update_messages_primary(self) -> None:
        """
        Update messages starting from the first graph
        """

        partial = self.weights.data.copy()
        partial += self.msg_h2n.sum(1).getA1()

        # Update messages from node to f
        self.msg_n2f.data[:] = partial
        self._update_factor_g_messages()
        self.msg_n2f.data += self.msg_g2n.data

        self.marginals.data[:] = self.msg_n2f.data

        # Update messages fron node to g
        self.msg_n2g.data[:] = partial
        self._update_factor_f_messages()
        self.msg_n2g.data += self.msg_f2n.data

        self.marginals.data += self.msg_f2n.data

    def _update_messages_secondary(self) -> None:
        """
        Update messages starting from the second graph
        """

        partial = self.weights.data.copy()
        partial += self.msg_h2n.sum(1).getA1()

        # Update messages from node to g
        self.msg_n2g.data[:] = partial
        self._update_factor_f_messages()
        self.msg_n2g.data += self.msg_f2n.data

        self.marginals.data[:] = self.msg_n2g.data

        # Update messages from node to f
        self.msg_n2f.data[:] = partial
        self._update_factor_g_messages()
        self.msg_n2f.data += self.msg_g2n.data

        self.marginals.data += self.msg_g2n.data

    def _round_messages(self) -> None:
        """
        Rounding phase
        """

        super(BeliefQAP, self)._round_messages()
        self._update_square_factor_messages()

    def _update_square_factor_messages(self) -> None:
        """

        Update the messages denoted by
        $$
        m_{h_{ii\\prime jj\\prime} \\rightarrow{} X_{ii\\prime}}
        $$

        The formula is the following one :
        $$
        m_{h_{ii\\prime j j\\prime} \\xrightarrow{} X_{ii\\prime}} = \\text{clip} (w_{ii\\prime jj\\prime} + m_{X_{jj\\prime}\\rightarrow{} h_{ii\\prime j j\\prime}}) - \\text{clip}(m_{X_{jj\\prime} \\xrightarrow{} h_{ii\\prime j j\\prime}})
        $$
        where $$ \\text{clip}(x) = max(0, x) $$
        """

        # partial is the message from node to square factor m(X[ii`] -> h[ii`jj`])
        partial = self.msg_h2n
        partial.data -= np.repeat(self.marginals.data, self.squares_per_edge)
        np.clip(partial.data, 0, partial.data.max(initial=0), out=partial.data)

        # transpose
        partial = partial.T.tocsr()

        self.msg_h2n.data[:] = self.weights_squares - partial.data
        np.clip(
            self.msg_h2n.data,
            0,
            self.msg_h2n.data.max(initial=0),
            out=self.msg_h2n.data,
        )
