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

import time
import numpy as np
import scipy
from scipy.sparse import csr_matrix, lil_matrix

from qbindiff.matcher import Matcher
from qbindiff.matcher.squares import find_squares
from qbindiff.utils import iter_csr_matrix


class TestSquares:
    """Regression tests for the squares matrix calculation"""

    def gen_rand_adj(self, size: int, density: float = 0.17):
        """Returns a square adjacency matrix with the specified density"""

        density = int(density * 100)
        mat = (np.random.randint(0, 100, size**2) < density).reshape((size, size))
        return mat

    def control_squares(self, primary_adj_matrix, secondary_adj_matrix, sparse_sim_matrix):
        """Slow but correct implementation of the square algorithm"""

        squares = []
        primary_children = []
        for node in primary_adj_matrix:
            primary_children.append([n for n, is_child in enumerate(node) if is_child])
        secondary_children = []
        for node in secondary_adj_matrix:
            secondary_children.append([n for n, is_child in enumerate(node) if is_child])

        for nodeA, nodeB, _ in iter_csr_matrix(sparse_sim_matrix):
            if len(primary_children[nodeA]) == 0 or len(secondary_children[nodeB]) == 0:
                continue
            for nodeC in secondary_children[nodeB]:
                for nodeD in primary_children[nodeA]:
                    if sparse_sim_matrix[nodeD, nodeC] > 0:
                        squares.append((nodeA, nodeB, nodeC, nodeD))

        return squares

    def control_squares_matrix(self, primary_adj_matrix, secondary_adj_matrix, sim_matrix):
        squares = self.control_squares(primary_adj_matrix, secondary_adj_matrix, sim_matrix)

        size = sim_matrix.nnz
        lil_squares_matrix = lil_matrix((size, size), dtype=np.uint8)
        # Give each similarity edge a unique number
        bipartite = sim_matrix.astype(np.uint32)
        bipartite.data[:] = np.arange(0, size, dtype=np.uint32)
        bipartite = bipartite.todok()

        # Populate the sparse squares matrix
        for nodeA, nodeB, nodeC, nodeD in squares:
            e1 = bipartite[nodeA, nodeB]
            e2 = bipartite[nodeD, nodeC]
            lil_squares_matrix[e1, e2] = 1
            lil_squares_matrix[e2, e1] = 1
        return lil_squares_matrix.tocsr()

    def test_algorithm_internal(self):
        """
        Test the optimized algorithm internally used by QBinDiff against the
        slow but surely working one.
        This is testing the algorithm for finding the squares
        """

        for k in range(10):
            primary_size = 30
            secondary_size = 50

            primary_adj_matrix = self.gen_rand_adj(primary_size, 0.2)
            secondary_adj_matrix = self.gen_rand_adj(secondary_size, 0.2)
            sim_matrix = scipy.sparse.rand(
                primary_size, secondary_size, density=0.1, dtype=np.float32
            ).tocsr()

            ret1 = set(find_squares(primary_adj_matrix, secondary_adj_matrix, sim_matrix))
            ret2 = set(self.control_squares(primary_adj_matrix, secondary_adj_matrix, sim_matrix))
            assert not (ret1 ^ ret2), "The optimized algorithm for finding the squares is faulty"

    def test_qbindiff_algorithm(self):
        """
        Test the optimized algorithm against the slow but surely working one.
        This is testing the whole sparse square matrix
        """

        for k in range(10):
            primary_size = 30
            secondary_size = 50

            primary_adj_matrix = self.gen_rand_adj(primary_size, 0.2)
            secondary_adj_matrix = self.gen_rand_adj(secondary_size, 0.2)
            sim_matrix = scipy.sparse.rand(
                primary_size, secondary_size, density=0.1, dtype=np.float32
            ).tocsr()

            matcher = Matcher(sim_matrix, primary_adj_matrix, secondary_adj_matrix)
            matcher.sparse_sim_matrix = sim_matrix

            matcher._compute_squares_matrix()
            correct = self.control_squares_matrix(
                primary_adj_matrix, secondary_adj_matrix, sim_matrix
            )

            assert not (matcher.squares_matrix != correct).max(), "The squares matrix is faulty"

    def test_performance(self):
        """Test the performance of the whole algorithm"""

        primary_size = 1000
        secondary_size = 1200
        time_limit = 10  # 10s

        primary_adj_matrix = self.gen_rand_adj(primary_size)
        secondary_adj_matrix = self.gen_rand_adj(secondary_size)
        sim_matrix = scipy.sparse.rand(
            primary_size, secondary_size, density=0.01, dtype=np.float32
        ).tocsr()

        matcher = Matcher(sim_matrix, primary_adj_matrix, secondary_adj_matrix)
        matcher.sparse_sim_matrix = sim_matrix

        start = time.time()
        matcher._compute_squares_matrix()
        end = time.time()

        assert end - start < time_limit, "Too slow to compute the squares matrix"
