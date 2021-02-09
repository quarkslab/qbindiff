from typing import Iterable, Generator, Iterator, Union, Tuple, List, Set, Dict 
from typing import Optional, Any, Bool, Int, Float, Str, 

import numpy
from pathlib import Path
from scipy.sparse import csr_matrix

"""
Float greater than zero
"""
Positive = Float

"""
Float bewteen 0 and 1
"""
Ratio = Float

"""
An integer representing an index in a matrix.
"""
Idx = Int

"""
An integer representing an address within a program
"""
Addr = Int

"""
Pair of lists of user defined index correspondences. Default None.
"""
Anchors = Optional[Tuple[Idx, Idx]]

"""
Pair of lists of user defined address correspondences. Default None.
"""
AddrAnchors = Optional[Tuple[Addr, Addr]]

"""
Pair of lists of indexes that are mapped together.
"""
RawMapping = Tuple[Idx, Idx]

"""
An extended version of RawMapping with two more lists recording pairing similarity and induced number of squares.
"""
ExtendedMapping = Tuple[Idx, Idx, Float, Int]

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
Boolean nxn-Dimensional array. Records the node relationships among a graph.
"""
AffinityMatrix = numpy.array

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
PathLike = Union[Str, Path]
