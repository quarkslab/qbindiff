from typing import Any

from qbindiff.loader import Program
from qbindiff.types import SimMatrix


def ZeroPass(
    sim_matrix: SimMatrix,
    primary: Program,
    secondary: Program,
    primary_mapping: dict[Any, int],
    secondary_mapping: dict[Any, int],
):
    """Set to zero all the -1 entries in the similarity matrix"""
    mask = sim_matrix == -1
    sim_matrix[mask] = 0
