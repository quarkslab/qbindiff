from qbindiff.features.mnemonic import MnemonicSimple, MnemonicTyped, GroupsCategory
from qbindiff.features.graph import (
    BBlockNb,
    MeanInsNB,
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
    DatName,
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
