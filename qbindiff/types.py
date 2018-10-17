from typing import Dict, Tuple, List, Optional, Set, Union, Generator, Iterator

import numpy
from pandas import DataFrame, Index
from scipy.sparse import csr_matrix

"""
An integer representing an address within a program
"""
Addr = int

"""
An integer representing an index in a matrix. At low-level
functions are manipulated using indexes in matrices.
"""
Idx = int

"""
Set of address indexes, as pandas Index
"""
AddrIndex = Index

"""
Low-level matching on matrix indexes computed by
the belief propagation algorithm. (Internal type)
"""
BeliefMatching = Generator[Tuple[Idx, Idx, float], None, None]

"""
1-Dimensional array. Use to represente slice of a matrix or function features
"""
Vector = numpy.array

"""
2-Dimensional float array. Represent the matrix of weights between functions
which represent similarity between functions
"""
SimMatrix = numpy.array

"""
Matrix based representation of the call graph. Boolean 2-D array
"""
CallMatrix = numpy.array

"""
2-D Matrix given to the belief propagation algorithm which represent distances
of both programs. It can be a DataFrame, a sparse matrix or a numpy array
"""
InputMatrix = Union[DataFrame, csr_matrix, numpy.array]

"""
Internal representation of call graphs, as a list of function indexes which
are parents to a list of indexes which are children
"""
CallGraph = List[List[Idx]]

"""
Math set of natural positive integers
"""
R = Union[int, float]

"""
Float bewteen 0 and 1
"""
Ratio = float

"""
Features of a function. The type is a dictionnary of features
keys to their occurence or value
"""
FunctionFeatures = Dict[str, int]

"""
Features extracted from the program. This a dictionnary indexed by function addreses
wich contains a dictionnary of features
"""
ProgramFeatures = Dict[Addr, FunctionFeatures]

"""
List of addresses of both programs to be anchored together.
For these function the match is fixed
"""
Anchors = Tuple[Optional[List[Addr]], Optional[List[Addr]]]
