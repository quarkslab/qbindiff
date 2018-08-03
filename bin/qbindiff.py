#!/usr/bin/env python3
# coding: utf-8

import click
from pathlib import Path
from tqdm import tqdm

# Disable this shitty matplotlib DEBUG messages !
import logging
logger = logging.getLogger('matplotlib')
logger.setLevel(logging.WARNING)


# Pop the directory of the script (to avoid import conflict with qbindiff (same name)
import sys
del sys.path[0]

from qbindiff.loader.program import Program
from qbindiff.features.mnemonic import MnemonicSimple, MnemonicTyped, GroupsCategory
from qbindiff.features.graph import GraphNbBlock, GraphMeanInstBlock, GraphMeanDegree, \
        GraphDensity, GraphNbComponents, GraphDiameter, GraphTransitivity, GraphCommunities
from qbindiff.features.artefact import LibName, DatName, Constant, ImpName
from qbindiff.differ.qbindiff import QBinDiff
from qbindiff.loader.types import LoaderType

LOADERS = list(x.name for x in LoaderType)
_FEATURES_TABLE = { MnemonicSimple.name: MnemonicSimple,
                    MnemonicTyped.name: MnemonicTyped,
                    GroupsCategory.name: GroupsCategory,
                    GraphNbBlock.name: GraphNbBlock,
                    GraphMeanInstBlock.name: GraphMeanInstBlock,
                    GraphMeanDegree.name: GraphMeanDegree,
                    GraphDensity.name: GraphDensity,
                    GraphNbComponents.name: GraphNbComponents,
                    GraphDiameter.name: GraphDiameter,
                    GraphTransitivity.name: GraphTransitivity,
                    GraphCommunities.name: GraphCommunities,
                    LibName.name: LibName,
                    DatName.name: DatName,
                    Constant.name: Constant,
                    ImpName.name: ImpName
                  }

FEATURES = list(_FEATURES_TABLE.keys())
DISTANCE = ['correlation', 'cosine']
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

def load_qbindiff_program(file_path):
    p_path = Path(file_path)
    p = Program(LoaderType.qbindiff, p_path / "data", p_path / "callgraph.json")
    logging.info("[+} %s loaded: %d functions" % (p.name, len(p)))
    return p


def load_binexport_program(file):
    p = Program(LoaderType.binexport, file)
    logging.info("[+} %s loaded: %d functions" % (p.name, len(p)))
    return p


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-o', '--output', type=click.Path(), default="matching.json", help="Output file matching")
@click.option('-l', '--loader', type=click.Choice(LOADERS), default=LoaderType.binexport.name, help="Input files type")
@click.option('-f', '--feature', type=click.Choice(FEATURES), default=None, multiple=True, help="Input files type")
@click.option('-d', '--distance', type=click.Choice(DISTANCE), default="correlation", help="Distance function to apply")
@click.option('-t', '--threshold', type=float, default=0.5, help="Distance treshold to keep matches [0.0 to 1.0]")
@click.option('-i', '--maxiter', type=int, default=50, help="Maximum number of iteration for belief propagation")
@click.option('--msim', type=int, default=1, help="Maximize similarity matching (alpha for NAQP)")
@click.option('--mcall', type=int, default=2, help='Maximize call graph matching (beta for NAQP)')
@click.option('--refine-match/--no-refine-match', default=True)
@click.argument("primary", type=click.Path(exists=True), metavar="<primary file>")
@click.argument('secondary', type=click.Path(exists=True), metavar="<secondary file>")
def main(output, loader, feature, distance, threshold, maxiter, msim, mcall, refine_match, primary, secondary):
    """
    qBinDiff is an experimental binary diffing tool based on
    machine learning technics, namely Belief propagation.
    """

    if 0.0 > threshold > 1:  # check the threshold value to fit
        logging.warning("Threshold value should within 0..1 (set it to 1.0)")
        threshold = 1
    # TODO: verify that alpha beta are positive

    # Preprocessing to extract features and filters functions
    logging.info("[+] loading programs")
    if loader == "qbindiff":
        p1 = load_qbindiff_program(primary)
        p2 = load_qbindiff_program(secondary)
    elif loader == "binexport":
        # checks here that selected features are supported
        unsupported_fts = [GroupsCategory.name]
        if [x for x in feature if x in unsupported_fts]:
            logging.warning("Useless feature %s for the binexport loader" % str(unsupported_fts))
        p1 = load_binexport_program(primary)
        p2 = load_binexport_program(secondary)
    else:
        logging.error("Diaphora loader not implemented yet..")
        exit(1)

    differ = QBinDiff(p1, p2)
    differ.distance = distance
    differ.maxiter = maxiter
    differ.threshold = threshold
    differ.alpha = msim
    differ.beta = mcall
    for name in feature:
        ft = _FEATURES_TABLE[name]()  # instanciate the feature
        differ.register_feature(ft)

    differ.initialize()
    logging.info("[+] starts NAQP computation")
    for niter in tqdm(differ.run(match_refine=refine_match), total=maxiter):
        pass

    differ.save_matching(output)
    exit(0)


if __name__ == '__main__':
    main()

