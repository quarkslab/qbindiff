from __future__ import annotations
from typing import Any, Iterable, TypeAlias, List, Tuple, Dict

import numpy
import enum
from pathlib import Path
from scipy.sparse import csr_matrix, csr_array
from collections import namedtuple

from qbindiff.abstract import GenericGraph

"""
Type of a feature value.
"""
FeatureValue: TypeAlias = float | Dict[str, float]

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
Anchors: TypeAlias = List[Tuple[Item, Item]]

"""
Pair of lists of indexes that are mapped together.
"""
RawMapping: TypeAlias = Tuple[List[Idx], List[Idx]]

"""
Match represent the matching between two functions and can hold the similarity between the two
"""
Match = namedtuple("Match", "primary secondary similarity confidence squares")


"""
An extended version of RawMapping with two more lists recording pairing similarity and induced number of squares.
"""
ExtendedMapping: TypeAlias = Iterable[Tuple[Item, Item, float, int]]

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
