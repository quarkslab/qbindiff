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
from qbindiff.features.artefact import Address, DatName, Constant
from qbindiff.features.topology import ChildNb, ParentNb, RelativeNb, LibName, ImpName
from qbindiff.features.wlgk import WeisfeilerLehman

FEATURES = {
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
    # BasicBlockFeatureExtractor
    # InstructionFeatureExtractor
    MnemonicSimple,
    MnemonicTyped,
    GroupsCategory,
    Address,
    DatName,
    # OperandFeatureExtractor
    Constant,
}

DEFAULT_FEATURES = {
    # FunctionFeatureExtractor
    BBlockNb,
    MeanInsNB,
    GraphMeanDegree,
    GraphDensity,
    GraphNbComponents,
    GraphDiameter,
    GraphTransitivity,
    ChildNb,
    ParentNb,
    RelativeNb,
    LibName,
    DatName,
    # BasicBlockFeatureExtractor
    # InstructionFeatureExtractor
    MnemonicSimple,
    MnemonicTyped,
    GroupsCategory,
    Address,
    DatName,
    # OperandFeatureExtractor
    Constant,
}
