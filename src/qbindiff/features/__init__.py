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
