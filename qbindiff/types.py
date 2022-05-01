from typing import Union, Tuple, List, Any, Iterable

import numpy
from pathlib import Path
from scipy.sparse import csr_matrix
from collections import namedtuple

from qbindiff.abstract import GenericGraph

"""
Float greater than zero
"""
Positive = float

"""
Float bewteen 0 and 1
"""
Ratio = float

"""
An integer representing an index in a matrix.
"""
Idx = int

"""
An integer representing an address within a program
"""
Addr = int

"""
Item, entity being matched. The only constraint is to be hashable
"""
Item = Any

"""
Pair of lists of user defined index correspondences. Default None.
"""
Anchors = List[Tuple[Item, Item]]

"""
Pair of lists of indexes that are mapped together.
"""
RawMapping = Tuple[List[Idx], List[Idx]]

"""
Match represent the matching between two functions and can hold the similarity between the two
"""
Match = namedtuple("Match", "primary secondary similarity squares")


"""
An extended version of RawMapping with two more lists recording pairing similarity and induced number of squares.
"""
ExtendedMapping = Iterable[Tuple[Item, Item, float, int]]

"""
Numpy data type
"""
Dtype = numpy.dtype

"""
Arbitrary d-Dimensional array. Used to represent a vector.
"""
Vector = numpy.array

"""
Arbitrary nxd-Dimensional array. Used to represent a matrix.
"""
Matrix = numpy.array

"""
Float nxd-Dimensional array. Each n rows is represented as a dimensionnal feature vector.
"""
FeatureVectors = numpy.array

"""
Boolean nxn-Dimensional array. It's the adjacency matrix representation of the graph.
"""
AdjacencyMatrix = numpy.array

"""
Float nxm-Dimensional array. Records the pairwise similarity scores between nodes of both graphs to diff.
"""
SimMatrix = numpy.array

"""
Float nxm-Dimensional array. A sparse version of the above SimMatrix
"""
SparseMatrix = csr_matrix

"""
Path
"""
PathLike = Union[str, Path]

"""
A generic Graph, iterable over the nodes
"""
Graph = GenericGraph
