#!/bin/env python3
"""
Python script for building documentation.
Usage
-----
    $ python make.py clean
    $ python make.py html
    $ python make.py latex
"""

import subprocess, os, sys
from pathlib import Path

DOC_PATH = Path(os.path.dirname(__file__))
SOURCE_PATH = DOC_PATH / "source"
BUILD_PATH = DOC_PATH / "build"
GOOD_BUILDERS = (
    "help",
    "html",
    "dirhtml",
    "singlehtml",
    "pickle",
    "json",
    "htmlhelp",
    "qthelp",
    "devhelp",
    "epub",
    "latex",
    "text",
    "man",
    "texinfo",
    "info",
    "gettext",
    "changes",
    "xml",
    "pseudoxml",
    "linkcheck",
    "doctest",
    "coverage",
    "clean",
)


def usage():
    print("Usage: python make.py BUILDER")


def sphinx_build(builder: str, njobs: int | None = None) -> int:
    """
    Call sphinx to build documentation.

    :param builder: the sphinx builder (html, latex, ...)
    :param njobs: Number of parallel jobs, if None the option is disabled
    """

    if builder not in GOOD_BUILDERS:
        raise ValueError(f"Unrecognized builder {builder}. Use one of {GOOD_BUILDERS}")

    cmd = ["sphinx-build", "-M", builder, SOURCE_PATH, BUILD_PATH]
    if njobs:
        cmd += ["-j", njobs]
    return subprocess.call(cmd)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage()
        exit(1)

    sphinx_build(sys.argv[1])
