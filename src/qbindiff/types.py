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

"""Contains all the type alias/definitions used by the module

This module contains all definitions of the type aliases and the generic enums
used by qbindiff.
"""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, TypeAlias, Protocol, TYPE_CHECKING

import numpy
from pathlib import Path
from scipy.sparse import csr_matrix, csr_array
from collections import namedtuple
import enum_tools.documentation
from enum import IntEnum

from qbindiff.abstract import GenericGraph

if TYPE_CHECKING:
    from qbindiff import Program

FeatureValue: TypeAlias = float | dict[str, float]
"""
Type of a feature value.
"""

Positive: TypeAlias = float
"""
Float greater than zero
"""

Ratio: TypeAlias = float
"""
Float bewteen 0 and 1
"""

Idx: TypeAlias = int
"""
An integer representing an index in a matrix.
"""

Addr: TypeAlias = int
"""
An integer representing an address within a program
"""

Item: TypeAlias = Any
"""
Item, entity being matched. The only constraint is to be hashable
"""

Anchors: TypeAlias = list[tuple[Item, Item]]
"""
Pair of lists of user defined index correspondences. Default None.
"""

RawMapping: TypeAlias = tuple[list[Idx], list[Idx]]
"""
Pair of lists of indexes that are mapped together.
"""

Match = namedtuple("Match", "primary secondary similarity confidence squares")
"""
Match represent the matching between two functions and can hold the similarity between the two
"""


ExtendedMapping: TypeAlias = Iterable[tuple[Item, Item, float, int]]
"""
An extended version of RawMapping with two more lists recording pairing similarity and induced number of squares.
"""

Dtype: TypeAlias = numpy.dtype
"""
Numpy data type
"""

Vector: TypeAlias = numpy.ndarray
"""
Arbitrary d-Dimensional array. Used to represent a vector.
"""

Matrix: TypeAlias = numpy.ndarray
"""
Arbitrary nxd-Dimensional array. Used to represent a matrix.
"""

FeatureVectors: TypeAlias = numpy.ndarray
"""
Float nxd-Dimensional array. Each n rows is represented as a dimensionnal feature vector.
"""

AdjacencyMatrix: TypeAlias = numpy.ndarray
"""
Boolean nxn-Dimensional array. It's the adjacency matrix representation of the graph.
"""

SimMatrix: TypeAlias = numpy.ndarray
"""
Float nxm-Dimensional array. Records the pairwise similarity scores between nodes of both graphs to diff.
"""

SparseMatrix: TypeAlias = csr_matrix
"""
Float nxm-Dimensional array. A sparse version of the above SimMatrix
"""

Graph: TypeAlias = GenericGraph
"""
A generic Graph, iterable over the nodes
"""

SparseVector: TypeAlias = csr_array
"""
Float n-Dimensional sparse array.
"""

PathLike: TypeAlias = str | Path
"""
Path
"""


class GenericPass(Protocol):
    """Callback function type for Passes"""

    def __call__(
        self,
        sim_matrix: SimMatrix,
        primary: Program,
        secondary: Program,
        primary_mapping: dict[Any, int],
        secondary_mapping: dict[Any, int],
        **kwargs,
    ) -> None:
        """Execute the pass that operates on the similarity matrix inplace"""
        raise NotImplementedError()


@enum_tools.documentation.document_enum
class Distance(IntEnum):
    """
    Enum of different (supported) distances used to compute the similarity matrix based on chosen features.
    """

    canberra = 0  # doc: canberra distance
    euclidean = 1  # doc: euclidean distance
    cosine = 2  # doc: cosine distance
    jaccard_strong = 3  # doc: custom distance
