#!/usr/bin/env python3
# coding: utf-8

import logging
import click
from pathlib import Path
from tqdm import tqdm
import coloredlogs
from coloredlogs import DEFAULT_FIELD_STYLES as FST, DEFAULT_LEVEL_STYLES as LST

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


def configure_logging(verbose):
    #first desactivate matplotlib logging
    logger = logging.getLogger('matplotlib')
    logger.setLevel(logging.WARNING)
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose >= 1 else logging.INFO)
    # coloredlogs.install(logger.level,
    #                     logger=logger,
    #                     fmt='[%(name)s] [%(levelname)s] %(message)s',
    #                     field_styles={**FST, **{'asctime': {'bold': True}, 'name': {'color': 'blue', 'bold': True}}},
    #                     level_styles={**LST, **{'debug': {'color': 'blue'}}})


def load_qbindiff_program(file_path):
    p_path = Path(file_path)
    p = Program(LoaderType.qbindiff, p_path / "data", p_path / "callgraph.json")
    logging.info("[+] %s loaded: %d functions" % (p.name, len(p)))
    return p


def load_binexport_program(file):
    p = Program(LoaderType.binexport, file)
    logging.info("[+] %s loaded: %d functions" % (p.name, len(p)))
    return p


LOADERS = list(x.name for x in LoaderType)
FEATURES = {MnemonicSimple,
            MnemonicTyped,
            GroupsCategory,
            GraphNbBlock,
            GraphMeanInstBlock,
            GraphMeanDegree,
            GraphDensity,
            GraphNbComponents,
            GraphDiameter,
            GraphTransitivity,
            GraphCommunities,
            LibName,
            DatName,
            Constant,
            ImpName
            # New features should be added here
            }

FEATURES_KEYS = {x.name: x for x in FEATURES}
FEATURES_KEYS.update({x.key: x for x in FEATURES})  # also add short keys as a valid feature
DISTANCE = ['correlation', 'cosine']

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'],
                        max_content_width=300)

DEFAULT_LOADER = LoaderType.binexport.name
DEFAULT_OUTPUT = "matching.json"
DEFAULT_DISTANCE = "correlation"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAXITER = 80
DEFAULT_ALPHA = 1
DEFAULT_BETA = 2

help_features = """The following features are available:
"""+''.join("- %s, %s: %s\n" % (x.key, x.name, x.__doc__) for x in FEATURES)

help_distance = "Mathematical distance function between cosine and correlation [default: %s]" % DEFAULT_DISTANCE


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-o', '--output', type=click.Path(), default=DEFAULT_OUTPUT, help="Output file matching [default: %s]" % DEFAULT_OUTPUT)
@click.option('-l', '--loader', type=click.Choice(LOADERS), default=DEFAULT_LOADER, metavar="<loader>",
              help="Input files type between %s. [default loader: %s]" % (LOADERS, DEFAULT_LOADER))
@click.option('-f', '--feature', type=click.Choice(FEATURES_KEYS), default=None, multiple=True, metavar="<feature>", help=help_features)
@click.option('-d', '--distance', type=click.Choice(DISTANCE), default=DEFAULT_DISTANCE, metavar="<function>", help=help_distance)
@click.option('-t', '--threshold', type=float, default=DEFAULT_THRESHOLD, help="Distance treshold to keep matches between 0.0 to 1.0 [default: %.02f]" % DEFAULT_THRESHOLD)
@click.option('-i', '--maxiter', type=int, default=DEFAULT_MAXITER, help="Maximum number of iteration for belief propagation [default: %d]" % DEFAULT_MAXITER)
@click.option('--msim', type=int, default=DEFAULT_ALPHA, help="Maximize similarity matching (alpha for NAQP) [default: %d]" % DEFAULT_ALPHA)
@click.option('--mcall', type=int, default=DEFAULT_BETA, help='Maximize call graph matching (beta for NAQP) [default: %d]' % DEFAULT_BETA)
@click.option('--refine-match/--no-refine-match', default=True)
@click.option('-v', '--verbose', count=True, help="Activate debugging messages")
@click.argument("primary", type=click.Path(exists=True), metavar="<primary file>")
@click.argument('secondary', type=click.Path(exists=True), metavar="<secondary file>")
def main(output, loader, feature, distance, threshold, maxiter, msim, mcall, refine_match, verbose, primary, secondary):
    """
    qBinDiff is an experimental binary diffing tool based on
    machine learning technics, namely Belief propagation.
    """

    configure_logging(verbose)

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
    registered_ft = set()
    for name in feature:
        ft = FEATURES_KEYS[name]
        if ft in registered_ft:
            logging.warning("feature %s already registered (skip it)" % (ft.name))
        else:
            differ.register_feature(ft())  # instanciate it
            registered_ft.add(ft)

    differ.initialize()
    logging.info("[+] starts NAQP computation")
    for niter in tqdm(differ.run(match_refine=refine_match), total=maxiter):
        pass

    differ.save_matching(output)
    exit(0)


if __name__ == '__main__':
    main()

