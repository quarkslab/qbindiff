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

"""Matcher
"""

# built-in imports
import logging

# Third-party imports
import numpy as np
from lapjv import lapjv
from scipy.sparse import csr_matrix, coo_matrix

# Local imports
from qbindiff.matcher.squares import find_squares
from qbindiff.matcher.belief_propagation import BeliefMWM, BeliefQAP
from qbindiff.types import (
    Positive,
    Ratio,
    RawMapping,
    AdjacencyMatrix,
    Matrix,
    SimMatrix,
)


def solve_linear_assignment(cost_matrix: Matrix) -> RawMapping:
    """
    Solve the linear assignment problem given the cost_matrix

    :param: cost matrix
    :return: raw mapping
    """

    n, m = cost_matrix.shape
    transposed = n > m
    if transposed:
        n, m = m, n
        cost_matrix = cost_matrix.T
    full_cost_matrix = np.zeros((m, m), dtype=cost_matrix.dtype)
    full_cost_matrix[:n, :m] = cost_matrix
    col_indices = lapjv(full_cost_matrix)[0][:n]
    if transposed:
        return col_indices, np.arange(n)
    return np.arange(n), col_indices


class Matcher:
    def __init__(
        self,
        similarity_matrix: SimMatrix,
        primary_adj_matrix: AdjacencyMatrix,
        secondary_adj_matrix: AdjacencyMatrix,
    ):
        self._mapping = None  # nodes mapping
        #: Similarity matrix used by the Matcher
        self.sim_matrix = similarity_matrix
        #: Adjacency matrix of the primary graph
        self.primary_adj_matrix = primary_adj_matrix
        #: Adjacency matrix of the secondary graph
        self.secondary_adj_matrix = secondary_adj_matrix

        self.sparse_sim_matrix = None
        self.squares_matrix = None

    def _compute_sparse_sim_matrix(self, sparsity_ratio: Ratio, sparse_row: bool) -> None:
        """
        Generate the sparse similarity matrix given the sparsity_ratio

        :param sparsity_ratio: ratio of least probable matches to ignore
        :param sparse_row: whether to use sparse rows
        :return: None
        """

        sparsity_size = round(sparsity_ratio * self.sim_matrix.size)

        if sparsity_size == 0:  # Keep the matrix as it is
            self.sparse_sim_matrix = csr_matrix(self.sim_matrix)
            return

        if sparsity_size == self.sim_matrix.size:  # Empty matrix
            self.sparse_sim_matrix = csr_matrix(self.sim_matrix.shape, dtype=self.sim_matrix.dtype)
            return

        if sparse_row:
            sparsity_size = round(sparsity_ratio * self.sim_matrix.shape[1])

            # Sort the similarity matrix columns and keep indexes of the sorted values
            sorted_indexes = np.argsort(self.sim_matrix, kind="stable")

            mask = []
            for i in range(self.sim_matrix.shape[0]):
                # Replace all the discarded values with zeros
                self.sim_matrix[i, sorted_indexes[i, :sparsity_size]] = 0

                # Append the mask for the current row
                mask.append(self.sim_matrix[i] > 0)

            self.sparse_sim_matrix = csr_matrix(mask, dtype=self.sim_matrix.dtype)
            self.sparse_sim_matrix.data[:] = self.sim_matrix[mask]
        else:
            # Sort the flattened similarity matrix and keep the indexes of the sorted values
            sorted_indexes = np.argsort(self.sim_matrix, axis=None, kind="stable")

            # Replace all the discarded values with zeros
            self.sim_matrix.flat[sorted_indexes[:sparsity_size]] = 0

            # Create a mask
            mask = self.sim_matrix > 0

            # Create the sparse matrix
            csr_data = self.sim_matrix[mask]
            self.sparse_sim_matrix = csr_matrix(mask, dtype=self.sim_matrix.dtype)
            self.sparse_sim_matrix.data[:] = csr_data

    def _compute_squares_matrix(self) -> None:
        """
        Generate the sparse squares matrix and store it in self._squares_matrix.
        Given two graphs G1 and G2, a square is a tuple of nodes (nodeA, nodeB, nodeC, nodeD)
        such that all of the followings statements are true:
          - nodeA and nodeD belong to G1
          - nodeB and nodeC belong to G2
          - (nodeA, nodeD) is a directed edge in G1
          - (nodeB, nodeC) is a directed edge in G2
          - (nodeA, nodeB) is a edge in the similarity matrix (non-zero score)
          - (nodeC, nodeD) is a edge in the similarity matrix (non-zero score)
        Note that the nodes are not necessarily different since (nodeX, nodeX) might be
        a valid edge.

        (A) <---sim_edge---> (B)
         |                    |
         |graph_edge          |
         |          graph_edge|
         v                    v
        (D) <---sim_edge---> (C)

        The resulting square matrix is stored as a csr_matrix of size NxN where N=#{similarity edge}
        Every similarity edge is given a unique increasing number from 0 to N and there is a square
        between two similarity edges `e1` and `e2` <=> (iff) self._squares_matrix[e1][e2] == 1

        The time complexity is O(|sparse_sim_matrix| * average_graph_degree**2)
        """

        # Use the fast Cython algorithm for efficiency
        self.squares_matrix = find_squares(
            self.primary_adj_matrix, self.secondary_adj_matrix, self.sparse_sim_matrix
        )

    @property
    def mapping(self) -> RawMapping:
        """
        Nodes mapping between the two graphs
        """
        return self._mapping

    @property
    def confidence_score(self) -> list[float]:
        """
        Confidence score for each match in the nodes mapping
        """
        return [self._confidence[idx1, idx2] for idx1, idx2 in zip(*self.mapping)]

    def process(
        self, sparsity_ratio: Ratio = 0.75, sparse_row: bool = False, compute_squares: bool = True
    ):
        """
        Initialize the matching algorithm

        :param sparsity_ratio: The ratio between null element over the entire similarity
                               matrix
        :param sparse_row: When building the sparse similarity matrix we can either
                           filter out the elements by considering all the entries in the
                           similarity matrix (sparse_row == False) or by considering
                           each vector separately (sparse_row == True)
        :param compute_squares: Whether to compute the squares matrix
        :return: None
        """

        logging.debug(
            f"Computing sparse similarity matrix (ratio {sparsity_ratio} sparse_row {sparse_row})"
        )
        self._compute_sparse_sim_matrix(sparsity_ratio, sparse_row)
        logging.debug(
            f"Sparse similarity matrix computed, shape: {self.sparse_sim_matrix.shape}"
            f", nnz elements: {self.sparse_sim_matrix.nnz}"
        )
        if compute_squares:
            logging.debug("Computing squares matrix")
            self._compute_squares_matrix()
            logging.debug(
                f"Squares matrix computed, shape: {self.squares_matrix.shape}"
                f", nnz elements: {self.squares_matrix.nnz}"
            )

    def compute(self, tradeoff: Ratio = 0.75, epsilon: Positive = 0.5, maxiter: int = 1000) -> None:
        """
        Launch the computation for a given number of iterations, using specific QBinDiff parameters

        :param tradeoff: tradeoff between the node similarity and the structure
        :param epsilon: perturbation to add to the similarity matrix
        :param maxiter: maximum number of iterations for the belief propagation
        :return: None
        """

        if tradeoff == 1:
            logging.info("[+] switching to Maximum Weight Matching (tradeoff is 1)")
            belief = BeliefMWM(self.sparse_sim_matrix, epsilon)
        else:
            belief = BeliefQAP(self.sparse_sim_matrix, self.squares_matrix, tradeoff, epsilon)

        for niter in belief.compute(maxiter):
            yield niter

        score_matrix = self.sparse_sim_matrix.copy()
        self._confidence = belief.current_marginals
        self._mapping = self.refine(belief.current_mapping, score_matrix)

    def refine(self, mapping: RawMapping, score_matrix: SimMatrix) -> RawMapping:
        """
        Refine the mappings between the nodes of the two graphs
        by matching the unassigned nodes

        :param mapping: initial mapping
        :param score_matrix: similarity matrix
        :return: updated raw mapping
        """

        primary, secondary = mapping
        assert len(primary) == len(secondary)

        # All the nodes have been assigned
        if len(primary) == min(score_matrix.shape):
            return mapping

        primary_missing = np.setdiff1d(range(score_matrix.shape[0]), primary)
        secondary_missing = np.setdiff1d(range(score_matrix.shape[1]), secondary)
        score_matrix = score_matrix[primary_missing][:, secondary_missing]
        nnz_indices = score_matrix.nonzero()
        score_matrix = score_matrix.toarray()
        # Give the zero elements a high score
        lap_scores = np.full(score_matrix.shape, 1000000, dtype=score_matrix.dtype)
        # LAP solves solves for the minimum cost but high scores means good match
        lap_scores[nnz_indices] = -score_matrix[nnz_indices]

        primary_ass, secondary_ass = solve_linear_assignment(lap_scores)

        return np.hstack((primary, primary_missing[primary_ass])), np.hstack(
            (secondary, secondary_missing[secondary_ass])
        )
