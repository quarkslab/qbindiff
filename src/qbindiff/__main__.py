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
from typing import TYPE_CHECKING

# Third-party imports
import rich_click as click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress
from rich.table import Table

# Local imports
from qbindiff import __version__ as qbindiff_version
from qbindiff import LoaderType, Program, QBinDiff, Mapping, Distance
from qbindiff.features import FEATURES, DEFAULT_FEATURES
from qbindiff.utils import log_once

if TYPE_CHECKING:
    from typing import Any


def configure_logging(verbose: int):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[RichHandler(rich_tracebacks=True, show_time=False)],
    )

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

    console = Console()

    table = Table(show_header=False)
    table.add_column(style="dim")
    table.add_column(justify="right")

    table.add_row("Score", "{:.4f}".format(similarity + nb_squares))
    table.add_row("Similarity", "{:.4f}".format(similarity))
    table.add_row("Squares", "{:.0f}".format(nb_squares))
    table.add_row("Nb matches", "{}".format(nb_matches), end_section=True)
    table.add_row(
        "Node cover",
        "{:.3f}% / {:.3f}%".format(
            100 * nb_matches / len(differ.primary_adj_matrix),
            100 * nb_matches / len(differ.secondary_adj_matrix),
        ),
    )
    table.add_row(
        "Edge cover",
        "{:.3f}% / {:.3f}%".format(
            100 * nb_squares / differ.primary_adj_matrix.sum(),
            100 * nb_squares / differ.secondary_adj_matrix.sum(),
        ),
    )

    console.print(table)


FEATURES_KEYS = {x.key: x for x in FEATURES}

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

DEFAULT_FEATURES = tuple(x.key for x in DEFAULT_FEATURES)
DEFAULT_DISTANCE = Distance.haussmann.name
DEFAULT_SPARSITY_RATIO = 0.6
DEFAULT_TRADEOFF = 0.8
DEFAULT_EPSILON = 0.9
DEFAULT_MAXITER = 1000
DEFAULT_OUTPUT = Path("qbindiff_results.csv")

# mapping from loader name to loader enum type
LOADERS = {x.name: x for x in LoaderType}


click.rich_click.SHOW_METAVARS_COLUMN = False
click.rich_click.APPEND_METAVARS_HELP = True
click.rich_click.STYLE_METAVAR_APPEND = "yellow"
click.rich_click.OPTION_GROUPS = {
    "qbindiff": [
        {"name": "Output parameters", "options": ["--output", "--format"]},
        {
            "name": "Primary file options",
            "options": ["--primary-loader", "--primary-executable", "--primary-arch"],
        },
        {
            "name": "Secondary file options",
            "options": ["--secondary-loader", "--secondary-executable", "--secondary-arch"],
        },
        {"name": "Global options", "options": ["--verbose", "--quiet", "--help", "--version"]},
        {
            "name": "Diffing parameters",
            "options": [
                "--feature",
                "--list-features",
                "--normalize",
                "--distance",
                "--tradeoff",
                "--sparsity-ratio",
                "--sparse-row",
                "--epsilon",
                "--maxiter",
            ],
        },
    ]
}


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


def load_program(
    name: str, loader_s: str, export: Path, exec_file: Path, arch: str = ""
) -> Program:
    if not loader_s:
        if export.suffix.casefold() == ".Quokka".casefold():
            loader_p = LOADERS["quokka"]
        elif export.suffix.casefold() == ".BinExport".casefold():
            loader_p = LOADERS["binexport"]
        else:
            logging.error(
                f"Cannot detect automatically the loader for the {name}, please specify it with `-l1`/`-l2`."
            )
            exit(1)
    else:
        loader_p = LOADERS[loader_s]

    # Check that the executables have been provided
    logging.info(f"[+] Loading {name}: {export.name}")
    if loader_p == LoaderType.quokka:
        if not (exec_file and os.path.exists(exec_file)):
            logging.error(
                "When using the quokka loader you have to provide the raw binaries (option `-e1`/`-e2`)."
            )
            exit(1)
        program = Program(export, exec_file, loader=loader_p)
    elif loader_p == LoaderType.ida:
        program = Program("", loader=loader_p)
    elif loader_p == LoaderType.binexport:
        program = Program(export, loader=loader_p, arch=arch)
    else:
        assert False
    return program


help_features = """\b
Features to use for the binary analysis, it can be specified multiple times.
Features may be weighted by a positive value (default 1.0) and/or compared with a
specific distance (by default the option -d is used) like this <feature>:<weight>:<distance>.
For a list of all the features available see --list-features."""


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-l1",
    "--primary-loader",
    "primary_loader",
    type=click.Choice(list(LOADERS.keys())),
    help=f"Enforce loader type.",
)
@click.option(
    "-l2",
    "--secondary-loader",
    "secondary_loader",
    type=click.Choice(list(LOADERS.keys())),
    help=f"Enforce loader type.",
)
@click.option(
    "-e1",
    "--primary-executable",
    "primary_exec",
    type=Path,
    help="Path to the raw executable (required for quokka exports).",
)
@click.option(
    "-e2",
    "--secondary-executable",
    "secondary_exec",
    type=Path,
    help="Path to the raw executable (required for quokka exports).",
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
    show_default=True,
    help="Normalize the Call Graph (can potentially lead to a partial matching).",
)
@click.option(
    "-d",
    "--distance",
    type=click.Choice([d.name for d in Distance]),
    show_default=True,
    default=DEFAULT_DISTANCE,
    help=f"Available distances:",
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
    help="Whether to build the sparse similarity matrix considering its entirety or processing it row per row.",
)
@click.option(
    "-t",
    "--tradeoff",
    type=float,
    show_default=True,
    default=DEFAULT_TRADEOFF,
    help=f"Tradeoff between function content (near 1.0) and call-graph information (near 0.0).",
)
@click.option(
    "-e",
    "--epsilon",
    type=float,
    show_default=True,
    default=DEFAULT_EPSILON,
    help=f"Relaxation parameter to enforce convergence.",
)
@click.option(
    "-i",
    "--maxiter",
    type=int,
    show_default=True,
    default=DEFAULT_MAXITER,
    help=f"Maximum number of iteration for belief propagation.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=True, dir_okay=False),
    default=DEFAULT_OUTPUT,
    show_default=True,
    help="Output file path.",
)
@click.option(
    "-ff",
    "--format",
    show_default=True,
    default="csv",
    type=click.Choice(["bindiff", "csv"]),
    help=f"Output file format.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    metavar="-v|-vv|-vvv",
    help="Activate debugging messages.",
)
@click.version_option(qbindiff_version)
@click.option(
    "-a1",
    "--primary-arch",
    type=str,
    help="Enforce disassembling architecture. Format is like 'CS_ARCH_X86:CS_MODE_64'.",
)
@click.option(
    "-a2",
    "--secondary-arch",
    type=str,
    help="Enforce disassembling architecture. Format is like 'CS_ARCH_X86:CS_MODE_64'.",
)
@click.option(
    "--list-features",
    is_flag=True,
    callback=list_features,
    expose_value=False,
    is_eager=True,
    help="List all the available features.",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Do not display progress bars and final statistics.",
)
@click.argument("primary", type=Path, metavar="<primary file>")
@click.argument("secondary", type=Path, metavar="<secondary file>")
def main(
    primary_loader,
    secondary_loader,
    features,
    normalize,
    distance,
    sparsity_ratio,
    sparse_row,
    tradeoff,
    epsilon,
    maxiter,
    primary_exec,
    secondary_exec,
    output,
    format,
    primary_arch,
    secondary_arch,
    quiet,
    verbose,
    primary,
    secondary,
):
    """
        QBinDiff is an experimental binary diffing tool based on
    machine learning technics, namely Belief propagation.

        Examples:

    - For Quokka exports:
    qbindiff -e1 file1.bin -e2 file2.bin file1.quokka file2.quokka

    - For BinExport exports, changing the output path:
    qbindiff -o my_diff.bindiff file1.BinExport file2.BinExport
    """

    configure_logging(verbose)

    if 0 > sparsity_ratio or sparsity_ratio > 1:
        logging.warning(
            "[-] Sparsity ratio should be within 0..1 (set it to %.2f)" % DEFAULT_SPARSITY_RATIO
        )
        sparsity_ratio = DEFAULT_SPARSITY_RATIO

    if 0 > tradeoff or tradeoff > 1:
        logging.warning(
            "[-] Trade-off parameter should be within 0..1 (set it to %.2f)" % DEFAULT_TRADEOFF
        )
        tradeoff = DEFAULT_TRADEOFF

    if 0 > epsilon:
        logging.warning(
            "[-] Epsilon parameter should be positive (set it to %.3f)" % DEFAULT_EPSILON
        )
        epsilon = DEFAULT_EPSILON

    with Progress() as progress:
        if not quiet:
            load_bar_total = 2
            load_bar = progress.add_task("File loading", total=load_bar_total)
            init_bar = progress.add_task("Initialization", start=False)
            match_bar = progress.add_task("Matching", start=False)
            save_bar_total = 1
            save_bar = progress.add_task("Saving Results", total=save_bar_total, start=False)

        # Check that the executables have been provided
        primary = load_program("primary", primary_loader, primary, primary_exec, primary_arch)
        progress.update(load_bar, advance=1) if not quiet else None

        # Check that the executables have been provided
        secondary = load_program(
            "secondary", secondary_loader, secondary, secondary_exec, secondary_arch
        )
        progress.update(load_bar, advance=1) if not quiet else None

        progress.start_task(init_bar) if not quiet else None
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
                qbindiff.register_feature_extractor(
                    extractor_class, float(weight), distance=distance
                )

        logging.info("[+] Initializing NAP")
        try:
            if not quiet:
                init_bar_total = 1000
                progress.update(init_bar, total=init_bar_total)
                prev_step = 0
                for step in qbindiff.process_iterator():
                    progress.update(init_bar, advance=step - prev_step)
                    prev_step = step
                progress.update(init_bar, completed=init_bar_total)
                progress.stop_task(init_bar)
                progress.start_task(match_bar)
            else:
                qbindiff.process()
        except TypeError as e:  # Catch ISA issue for binexport
            progress.stop()
            log_once(logging.ERROR, str(e))
            exit(1)

        logging.info("[+] Computing NAP")
        if not quiet:
            match_bar_total = qbindiff.maxiter
            progress.update(match_bar, total=match_bar_total)
            for _ in qbindiff.matching_iterator():
                progress.update(match_bar, advance=1)
            progress.update(match_bar, completed=match_bar_total)
            progress.stop_task(match_bar)
            progress.start_task(save_bar)
        else:
            qbindiff.compute_matching()

        logging.info("[+] Saving")
        if format == "bindiff":
            qbindiff.export_to_bindiff(output)
        elif format == "csv":
            qbindiff.mapping.to_csv(output, ("name", lambda f: f.name))
        logging.info("[+] Mapping successfully saved to: %s" % output)
        if not quiet:
            progress.update(save_bar, advance=save_bar_total)
            progress.stop_task(save_bar)
    if not quiet:
        display_statistics(qbindiff, qbindiff.mapping)


if __name__ == "__main__":
    main()
