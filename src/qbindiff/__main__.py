#!/usr/bin/env python3
"""
Copyright 2023 Quarkslab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# builtin-imports
from __future__ import annotations
import logging
import os
from pathlib import Path
from collections import defaultdict
from typing import TYPE_CHECKING

# Third-party imports
import click

# Local imports
from qbindiff import __version__ as qbindiff_version
from qbindiff import LoaderType, Program, QBinDiff, Mapping, Distance
from qbindiff.loader import LOADERS
from qbindiff.features import FEATURES, DEFAULT_FEATURES

if TYPE_CHECKING:
    from typing import Any


def configure_logging(verbose: int):
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

    logger = logging.getLogger()
    if verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)


def display_statistics(differ: QBinDiff, mapping: Mapping) -> None:
    nb_matches = mapping.nb_match
    similarity = mapping.similarity
    nb_squares = mapping.squares

    output = (
        "Score: {:.4f} | "
        "Similarity: {:.4f} | "
        "Squares: {:.0f} | "
        "Nb matches: {}\n".format(similarity + nb_squares, similarity, nb_squares, nb_matches)
    )
    output += "Node cover:  {:.3f}% / {:.3f}% | " "Edge cover:  {:.3f}% / {:.3f}%\n".format(
        100 * nb_matches / len(differ.primary_adj_matrix),
        100 * nb_matches / len(differ.secondary_adj_matrix),
        100 * nb_squares / differ.primary_adj_matrix.sum(),
        100 * nb_squares / differ.secondary_adj_matrix.sum(),
    )
    print(output)


FEATURES_KEYS = {x.key: x for x in FEATURES}

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], max_content_width=120)

DEFAULT_FEATURES = tuple(x.key for x in DEFAULT_FEATURES)
DEFAULT_DISTANCE = Distance.canberra.name
DEFAULT_SPARSITY_RATIO = 0.75
DEFAULT_TRADEOFF = 0.75
DEFAULT_EPSILON = 0.5
DEFAULT_MAXITER = 1000

LOADERS_KEYS = list(LOADERS.keys())


def list_features(ctx: click.Context, param: click.Parameter, value: Any) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.echo("The following features are available:")
    for feature in FEATURES:
        click.echo(f"  - {feature.key}:")
        click.echo(f"    {feature.help_msg}\n")
    # Also add the 'all' option
    click.echo(f"  - all: Enable every working features\n")
    ctx.exit()


help_features = """\b
Features to use for the binary analysis, it can be specified multiple times.
Features may be weighted by a positive value (default 1.0) and/or compared with a
specific distance (by default the option -d is used) like this <feature>:<weight>:<distance>.
For a list of all the features available see --list-features."""


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-l1",
    "--loader1",
    "loader_primary",
    type=click.Choice(LOADERS_KEYS),
    show_default=True,
    default="binexport",
    metavar="<loader>",
    help=f"Loader type to be used for the primary. Must be one of these {LOADERS_KEYS}",
)
@click.option(
    "-l2",
    "--loader2",
    "loader_secondary",
    type=click.Choice(LOADERS_KEYS),
    show_default=True,
    default="binexport",
    metavar="<loader>",
    help=f"Loader type to be used for the secondary. Must be one of these {LOADERS_KEYS}",
)
@click.option(
    "-f",
    "--feature",
    "features",
    type=str,
    default=DEFAULT_FEATURES,
    multiple=True,
    metavar="<feature>",
    help=help_features,
)
@click.option(
    "-n",
    "--normalize",
    is_flag=True,
    help="Normalize the Call Graph (can potentially lead to a partial matching). [default disabled]",
)
@click.option(
    "-d",
    "--distance",
    type=click.Choice((d.name for d in Distance)),
    show_default=True,
    default=DEFAULT_DISTANCE,
    metavar="<function>",
    help=f"The following distances are available {[d.name for d in Distance]}",
)
@click.option(
    "-s",
    "--sparsity-ratio",
    type=float,
    show_default=True,
    default=DEFAULT_SPARSITY_RATIO,
    help=f"Ratio of least probable matches to ignore. Between 0.0 (nothing is ignored) to 1.0 (only perfect matches are considered)",
)
@click.option(
    "-sr",
    "--sparse-row",
    is_flag=True,
    help="Whether to build the sparse similarity matrix considering its entirety or processing it row per row",
)
@click.option(
    "-t",
    "--tradeoff",
    type=float,
    show_default=True,
    default=DEFAULT_TRADEOFF,
    help=f"Tradeoff between function content (near 1.0) and call-graph information (near 0.0)",
)
@click.option(
    "-e",
    "--epsilon",
    type=float,
    show_default=True,
    default=DEFAULT_EPSILON,
    help=f"Relaxation parameter to enforce convergence",
)
@click.option(
    "-i",
    "--maxiter",
    type=int,
    show_default=True,
    default=DEFAULT_MAXITER,
    help=f"Maximum number of iteration for belief propagation",
)
@click.option(
    "-e1",
    "--executable1",
    "exec_primary",
    type=Path,
    help="Path to the primary raw executable. Must be provided if using quokka loader",
)
@click.option(
    "-e2",
    "--executable2",
    "exec_secondary",
    type=Path,
    help="Path to the secondary raw executable. Must be provided if using quokka loader",
)
@click.option(
    "-o",
    "--output",
    type=Path,
    help="Write output to PATH",
)
@click.option(
    "-ff",
    "--file-format",
    show_default=True,
    default="bindiff",
    type=click.Choice(["bindiff"]),
    help=f"The file format of the output file. Supported formats are [bindiff]",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Activate debugging messages. Can be supplied multiple times to increase verbosity",
)
@click.version_option(qbindiff_version)
@click.option(
    "--arch-primary",
    type=str,
    help="Force the architecture when disassembling for the primary. Format is 'CS_ARCH_X:CS_MODE_Ya,CS_MODE_Yb,...'",
)
@click.option(
    "--arch-secondary",
    type=str,
    help="Force the architecture when disassembling for the secondary. Format is 'CS_ARCH_X:CS_MODE_Ya,CS_MODE_Yb,...'",
)
@click.option(
    "--list-features",
    is_flag=True,
    callback=list_features,
    expose_value=False,
    is_eager=True,
    help="List all the available features",
)
@click.argument("primary", type=Path, metavar="<primary file>")
@click.argument("secondary", type=Path, metavar="<secondary file>")
def main(
    loader_primary,
    loader_secondary,
    features,
    normalize,
    distance,
    sparsity_ratio,
    sparse_row,
    tradeoff,
    epsilon,
    maxiter,
    exec_primary,
    exec_secondary,
    output,
    file_format,
    arch_primary,
    arch_secondary,
    verbose,
    primary,
    secondary,
):
    """
    QBinDiff is an experimental binary diffing tool based on
    machine learning technics, namely Belief propagation.
    """

    configure_logging(verbose)

    if 0.0 > sparsity_ratio > 1:
        logging.warning(
            "[-] Sparsity ratio should be within 0..1 (set it to %.2f)" % DEFAULT_SPARSITY_RATIO
        )
        sparsity_ratio = DEFAULT_SPARSITY_RATIO

    if 0.0 > tradeoff > 1:
        logging.warning(
            "[-] Trade-off parameter should be within 0..1 (set it to %.2f)" % DEFAULT_TRADEOFF
        )
        tradeoff = DEFAULT_TRADEOFF

    if 0.0 > epsilon:
        logging.warning(
            "[-] Epsilon parameter should be positive (set it to %.3f)" % DEFAULT_EPSILON
        )
        epsilon = DEFAULT_EPSILON

    if not output:
        logging.warning("[-] You have not specified an output file")

    loader_p = LOADERS[loader_primary]
    loader_s = LOADERS[loader_secondary]

    # Check that the executables have been provided
    if loader_p == LoaderType.quokka:
        if not (exec_primary and os.path.exists(exec_primary)):
            logging.error("When using the quokka loader you have to provide the raw binaries")
            exit(1)
        logging.info(f"[+] Loading primary: {primary.name}")
        primary = Program(loader_p, primary, exec_primary)
    elif loader_p == LoaderType.ida:
        logging.info(f"[+] Loading primary: {primary.name}")
        primary = Program(loader_p, primary)
    else:
        # BinExport
        logging.info(f"[+] Loading primary: {primary.name}")
        primary = Program(loader_p, primary, arch=arch_primary)

    # Check that the executables have been provided
    if loader_s == LoaderType.quokka:
        if not (exec_secondary and os.path.exists(exec_secondary)):
            logging.error("When using the quokka loader you have to provide the raw binaries")
            exit(1)
        logging.info(f"[+] Loading secondary: {secondary.name}")
        secondary = Program(loader_s, secondary, exec_secondary)
    elif loader_p == LoaderType.ida:
        logging.info(f"[+] Loading secondary: {secondary.name}")
        secondary = Program(loader_p, secondary)
    else:
        # BinExport
        logging.info(f"[+] Loading secondary: {secondary.name}")
        secondary = Program(loader_s, secondary, arch=arch_secondary)

    try:
        qbindiff = QBinDiff(
            primary,
            secondary,
            sparsity_ratio=sparsity_ratio,
            tradeoff=tradeoff,
            epsilon=epsilon,
            distance=Distance[distance],
            maxiter=maxiter,
            normalize=normalize,
            sparse_row=sparse_row,
        )
    except Exception as e:
        logging.error(e)
        exit(1)

    if not features:
        logging.error("no feature provided")
        exit(1)

    # Check for the 'all' option
    if "all" in set(features):
        # Add all features with default weight and distance
        for f in FEATURES_KEYS:
            qbindiff.register_feature_extractor(
                FEATURES_KEYS[f], float(1.0), distance=Distance[distance]
            )
    else:
        for feature in set(features):
            weight = 1.0
            distance = None
            if ":" in feature:
                feature, *opts = feature.split(":")
                if len(opts) == 2:
                    weight, distance = opts
                elif len(opts) == 1:
                    try:
                        weight = float(opts[0])
                    except ValueError:
                        distance = opts[0]
                else:
                    logging.error(f"Malformed feature {feature}")
                    continue
            if feature not in FEATURES_KEYS:
                logging.warning(f"Feature '{feature}' not recognized - ignored.")
                continue
            extractor_class = FEATURES_KEYS[feature]
            if distance is not None:
                distance = Distance[distance]
            qbindiff.register_feature_extractor(extractor_class, float(weight), distance=distance)

    logging.info("[+] Initializing NAP")
    qbindiff.process()

    logging.info("[+] Computing NAP")
    qbindiff.compute_matching()

    display_statistics(qbindiff, qbindiff.mapping)

    if output:
        logging.info("[+] Saving")
        if file_format == "bindiff":
            qbindiff.export_to_bindiff(output)
        logging.info("[+] Mapping successfully saved to: %s" % output)


if __name__ == "__main__":
    main()
