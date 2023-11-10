# Copyright 2023 Quarkslab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qbindiff.loader import Program
    from qbindiff.types import SimMatrix, FeatureValue, Addr, Idx
    from qbindiff.features.extractor import FeatureCollector


def ZeroPass(
    sim_matrix: SimMatrix,
    primary: Program,
    secondary: Program,
    primary_mapping: dict[Addr, Idx],
    secondary_mapping: dict[Addr, Idx],
    primary_features: dict[Addr, FeatureCollector],
    secondary_features: dict[Addr, FeatureCollector],
) -> None:
    """Set to zero all the -1 entries in the similarity matrix"""
    mask = sim_matrix == -1
    sim_matrix[mask] = 0
