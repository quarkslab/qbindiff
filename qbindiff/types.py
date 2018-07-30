from typing import Dict, Tuple, List, Optional, Set, Union, Generator

import numpy
from pandas import DataFrame, Index
from scipy.sparse import csr_matrix


from qbindiff.loader.program import Program
from qbindiff.features.visitor import ProgramVisitor

Addr = int
Idx = int
AddrIndex = Index  # panda index of addresses
Matching = Dict[Addr, Addr]
BeliefMatching = List[Tuple[Idx, Optional[Idx]]]
FinalMatching = List[Tuple[Optional[Addr], Optional[Addr]]]
Vector = numpy.array  # 1-Dimensional array
InputMatrix = Union[DataFrame, csr_matrix, numpy.array]
CallGraph = List[List[Idx]]
ℝ = Union[int, float]
