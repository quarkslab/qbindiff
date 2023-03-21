from qbindiff.features.mnemonic import MnemonicSimple, MnemonicTyped, GroupsCategory
from qbindiff.features.graph import (
    BBlockNb,
    StronglyConnectedComponents,
    BytesHash
    CyclomaticComplexity,
    MDIndex,
    SmallPrimeNumbers, 
    ReadWriteAccess,
    MeanInsNB,
    GraphMeanDegree,
    GraphDensity,
    GraphNbComponents,
    GraphDiameter,
    GraphTransitivity,
    GraphCommunities,
)

#TODO : other features from graph are not included (ex: JumpNb) => why ?

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
    ReadWriteAccess,
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
    MnemonicSimple,
    MnemonicTyped,
    GroupsCategory,
    Address,
    DatName,
    StrRef,
    # OperandFeatureExtractor
    Constant,
)

DEFAULT_FEATURES = (
    WeisfeilerLehman,
    FuncName,
    Address,
    DatName,
    Constant,
)
