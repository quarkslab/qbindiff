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

"""Collection of features

The module contains a collection of features that can be used to extract
relevant data to compute the similarity matrix.
Features are heuristics that operate at specific level inside the function
(operand, instruction, basic block, function) to extract a feature vector.
You can also think of feature vectors as a way of compressing the information
that was extracted in a mathematical object.

This information should help to characterize the function.
"""

from qbindiff.features.mnemonic import MnemonicSimple, MnemonicTyped, GroupsCategory

from qbindiff.features.graph import (
    BBlockNb,
    StronglyConnectedComponents,
    BytesHash,
    CyclomaticComplexity,
    MDIndex,
    JumpNb,
    SmallPrimeNumbers,
    ReadWriteAccess,
    MaxParentNb,
    MaxChildNb,
    MaxInsNB,
    MeanInsNB,
    InstNB,
    GraphMeanDegree,
    GraphDensity,
    GraphNbComponents,
    GraphDiameter,
    GraphTransitivity,
    GraphCommunities,
)

from qbindiff.features.artefact import Address, DatName, Constant, FuncName, StrRef
from qbindiff.features.topology import ChildNb, ParentNb, RelativeNb, LibName, ImpName
from qbindiff.features.wlgk import WeisfeilerLehman

FEATURES = (
    # FunctionFeatureExtractor
    BBlockNb,
    StronglyConnectedComponents,
    BytesHash,
    CyclomaticComplexity,
    MDIndex,
    SmallPrimeNumbers,
    MeanInsNB,
    GraphMeanDegree,
    GraphDensity,
    GraphNbComponents,
    GraphDiameter,
    GraphTransitivity,
    GraphCommunities,
    ChildNb,
    ParentNb,
    RelativeNb,
    LibName,
    WeisfeilerLehman,
    FuncName,
    # BasicBlockFeatureExtractor
    # InstructionFeatureExtractor
    JumpNb,
    MnemonicSimple,
    MnemonicTyped,
    GroupsCategory,
    Address,
    DatName,
    StrRef,
    # OperandFeatureExtractor
    ReadWriteAccess,
    Constant,
)

DEFAULT_FEATURES = (
    WeisfeilerLehman,
    FuncName,
    Address,
    DatName,
    Constant,
)
