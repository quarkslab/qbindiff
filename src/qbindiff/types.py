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

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, TypeAlias

import numpy
from pathlib import Path
from scipy.sparse import csr_matrix, csr_array
from collections import namedtuple
import enum_tools.documentation
from enum import IntEnum

from qbindiff.abstract import GenericGraph

"""
Type of a feature value.
"""
FeatureValue: TypeAlias = float | dict[str, float]

"""
Float greater than zero
"""
Positive: TypeAlias = float

"""
Float bewteen 0 and 1
"""
Ratio: TypeAlias = float

"""
An integer representing an index in a matrix.
"""
Idx: TypeAlias = int

"""
An integer representing an address within a program
"""
Addr: TypeAlias = int

"""
Item, entity being matched. The only constraint is to be hashable
"""
Item: TypeAlias = Any

"""
Pair of lists of user defined index correspondences. Default None.
"""
Anchors: TypeAlias = list[tuple[Item, Item]]

"""
Pair of lists of indexes that are mapped together.
"""
RawMapping: TypeAlias = tuple[list[Idx], list[Idx]]

"""
Match represent the matching between two functions and can hold the similarity between the two
"""
Match = namedtuple("Match", "primary secondary similarity confidence squares")


"""
An extended version of RawMapping with two more lists recording pairing similarity and induced number of squares.
"""
ExtendedMapping: TypeAlias = Iterable[tuple[Item, Item, float, int]]

"""
Numpy data type
"""
Dtype: TypeAlias = numpy.dtype

"""
Arbitrary d-Dimensional array. Used to represent a vector.
"""
Vector: TypeAlias = numpy.array

"""
Arbitrary nxd-Dimensional array. Used to represent a matrix.
"""
Matrix: TypeAlias = numpy.array

"""
Float nxd-Dimensional array. Each n rows is represented as a dimensionnal feature vector.
"""
FeatureVectors: TypeAlias = numpy.array

"""
Boolean nxn-Dimensional array. It's the adjacency matrix representation of the graph.
"""
AdjacencyMatrix: TypeAlias = numpy.array

"""
Float nxm-Dimensional array. Records the pairwise similarity scores between nodes of both graphs to diff.
"""
SimMatrix: TypeAlias = numpy.array

"""
Float nxm-Dimensional array. A sparse version of the above SimMatrix
"""
SparseMatrix: TypeAlias = csr_matrix

"""
Float n-Dimensional sparse array.
"""
SparseVector: TypeAlias = csr_array

"""
Path
"""
PathLike: TypeAlias = str | Path

"""
A generic Graph, iterable over the nodes
"""
Graph: TypeAlias = GenericGraph


@enum_tools.documentation.document_enum
class Distance(IntEnum):
    """
    Enum of different (supported) distances used to compute the similarity matrix based on chosen features.
    """

    canberra = 0  # doc: canberra distance
    euclidean = 1  # doc: euclidean distance
    cosine = 2  # doc: cosine distance
    jaccard_strong = 3  # doc: custom distance
