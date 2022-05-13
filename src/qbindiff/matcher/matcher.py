# built-in imports
import logging

# Third-party imports
import numpy as np
from lapjv import lapjv
from scipy.sparse import csr_matrix, lil_matrix

# Local imports
from qbindiff.matcher.belief_propagation import BeliefMWM, BeliefQAP
from qbindiff.types import (
    Positive,
    Ratio,
    RawMapping,
    AdjacencyMatrix,
    Matrix,
    SimMatrix,
    SparseMatrix,
)


def iter_csr_matrix(matrix: SparseMatrix):
    """
    Iter over non-null items in a CSR (Compressed Sparse Row) matrix.
    It returns a generator that, after each iteration, returns the tuple (row_index, column_index, value)
    """
    coo_matrix = matrix.tocoo()
    for x, y, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
        yield (x, y, v)


def solve_linear_assignment(cost_matrix: Matrix) -> RawMapping:
    """Solve the linear assignment problem given the cost_matrix"""
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
        self.sim_matrix = similarity_matrix
        self.primary_adj_matrix = primary_adj_matrix
        self.secondary_adj_matrix = secondary_adj_matrix

        self.sparse_sim_matrix = None
        self.squares_matrix = None

    def _compute_sparse_sim_matrix(self, sparsity_ratio: Ratio):
        """Generate the sparse similarity matrix given the sparsity_ratio"""
        ratio = round(sparsity_ratio * self.sim_matrix.size)

        if ratio == 0:
            self.sparse_sim_matrix = csr_matrix(self.sim_matrix)
            return
        elif ratio == self.sim_matrix.size:
            threshold = self.sim_matrix.max(1, keepdims=True)
            self.sparse_sim_matrix = self.sim_matrix >= threshold
            return

        threshold = np.partition(self.sim_matrix, ratio - 1, axis=None)[ratio]
        # We never want to match nodes with a similarity score of 0, even if that's the
        # right threshold
        if threshold == 0:
            threshold += 1e-8
        mask = self.sim_matrix >= threshold
        csr_data = self.sim_matrix[mask]

        self.sparse_sim_matrix = csr_matrix(mask, dtype=self.sim_matrix.dtype)
        self.sparse_sim_matrix.data[:] = csr_data

    def _compute_squares_matrix(self):
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

        squares = []
        primary_children = []
        for node in self.primary_adj_matrix:
            primary_children.append([n for n, is_child in enumerate(node) if is_child])
        secondary_children = []
        for node in self.secondary_adj_matrix:
            secondary_children.append(
                [n for n, is_child in enumerate(node) if is_child]
            )

        for nodeA, nodeB, score in iter_csr_matrix(self.sparse_sim_matrix):
            if len(primary_children[nodeA]) == 0 or len(secondary_children[nodeB]) == 0:
                continue
            for nodeC in secondary_children[nodeB]:
                for nodeD in primary_children[nodeA]:
                    if self.sparse_sim_matrix[nodeD, nodeC] > 0:
                        squares.append((nodeA, nodeB, nodeC, nodeD))

        size = self.sparse_sim_matrix.nnz
        lil_squares_matrix = lil_matrix((size, size), dtype=np.uint8)
        # Give each similarity edge a unique number
        bipartite = self.sparse_sim_matrix.astype(np.uint32)
        bipartite.data[:] = np.arange(0, size, dtype=np.uint32)

        # Populate the sparse squares matrix
        for nodeA, nodeB, nodeC, nodeD in squares:
            e1 = bipartite[nodeA, nodeB]
            e2 = bipartite[nodeD, nodeC]
            lil_squares_matrix[e1, e2] = 1
            lil_squares_matrix[e2, e1] = 1
        self.squares_matrix = lil_squares_matrix.tocsr()

    @property
    def mapping(self) -> RawMapping:
        """Returns the nodes mapping between the two graphs"""
        return self._mapping

    def process(self, sparsity_ratio: Ratio = 0.75, compute_squares: bool = True):
        self._compute_sparse_sim_matrix(sparsity_ratio)
        if compute_squares:
            self._compute_squares_matrix()

    def compute(
        self, tradeoff: Ratio = 0.75, epsilon: Positive = 0.5, maxiter: int = 1000
    ):
        if tradeoff == 1:
            logging.info("[+] switching to Maximum Weight Matching (tradeoff is 1)")
            belief = BeliefMWM(self.sparse_sim_matrix, epsilon)
        else:
            belief = BeliefQAP(
                self.sparse_sim_matrix, self.squares_matrix, tradeoff, epsilon
            )

        for niter in belief.compute(maxiter):
            yield niter

        score_matrix = self.sparse_sim_matrix.copy()
        self._mapping = self.refine(belief.current_mapping, score_matrix)

    def refine(self, mapping: RawMapping, score_matrix: SimMatrix) -> RawMapping:
        """
        Refine the mappings between the nodes of the two graphs
        by matching the unassigned nodes
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
        # Give the zero elements a high score
        lap_scores = np.full(score_matrix.shape, 1000000, dtype=score_matrix.dtype)
        # LAP solves solves for the minimum cost but high scores means good match
        lap_scores[nnz_indices] = -score_matrix[nnz_indices]

        primary_ass, secondary_ass = solve_linear_assignment(lap_scores)

        return np.hstack((primary, primary_missing[primary_ass])), np.hstack(
            (secondary, secondary_missing[secondary_ass])
        )
