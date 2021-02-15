from qbindiff.features.mnemonic import MnemonicSimple, MnemonicTyped, GroupsCategory
from qbindiff.features.graph import BBlockNb, MeanInsNB, GraphMeanDegree, \
                                    GraphDensity, GraphNbComponents, GraphDiameter, GraphTransitivity, GraphCommunities
from qbindiff.features.artefact import LibName, DatName, Constant, ImpName, Address
from qbindiff.features.topology import ChildNb, ParentNb, RelativeNb

FEATURES = {MnemonicSimple,
            MnemonicTyped,
            GroupsCategory,
            BBlockNb,
            MeanInsNB,
            GraphMeanDegree,
            GraphDensity,
            GraphNbComponents,
            GraphDiameter,
            GraphTransitivity,
            GraphCommunities,
            LibName,
            DatName,
            Constant,
            ImpName,
            ChildNb,
            ParentNb,
            RelativeNb,
            Address
            # New features should be added here
            }
