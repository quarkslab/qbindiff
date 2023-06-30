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

import pytest
import numpy as np
from qbindiff.matcher import Matcher


@pytest.mark.parametrize(
    "sparsity_ratio,sparse_row",
    (
        (0.1, False),
        (0.8, False),
        (0, False),
        (1, False),
        (0.1, True),
        (0.8, True),
        (0, True),
        (1, True),
    ),
)
def test_sparse_matrix(sparsity_ratio: float, sparse_row: bool):
    """Test the sparse matrix algorithm"""

    # Build the test similarity matrix
    np.random.seed(0)
    matrix = np.random.random((100, 100))
    m = Matcher(matrix, None, None)
    m._compute_sparse_sim_matrix(sparsity_ratio, sparse_row)

    # Check the algorithm
    assert m.sparse_sim_matrix is not None

    nnz_elements = len(matrix.nonzero()[0])

    if sparsity_ratio == 0:
        assert nnz_elements == m.sparse_sim_matrix.nnz
    if sparsity_ratio == 1:
        assert m.sparse_sim_matrix.nnz == 0

    sparse_size = round(sparsity_ratio * matrix.size)
    if sparse_row:
        sparse_size = round(sparsity_ratio * matrix.shape[1])
        for k in range(matrix.shape[0]):
            assert matrix.shape[1] - sparse_size == m.sparse_sim_matrix[k].nnz
    else:
        assert matrix.size - sparse_size == m.sparse_sim_matrix.nnz
