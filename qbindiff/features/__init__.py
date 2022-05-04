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
from qbindiff.features.artefact import Address
from qbindiff.features.topology import ChildNb, ParentNb, RelativeNb

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
    # BasicBlockFeatureExtractor
    # InstructionFeatureExtractor
    MnemonicSimple,
    MnemonicTyped,
    GroupsCategory,
    Address,
    # OperandFeatureExtractor
    # ~ LibName,
    # ~ DatName,
    # ~ Constant,
    # ~ ImpName,
}
