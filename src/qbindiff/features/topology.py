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

"""Features that uses the topology of the CG

These features leverage the call graph topology.

Use with care as the the topology of the call graph is also used in the belief
propagation algorithm, so using these features will result in relying
excessively on that information.
"""

from collections import defaultdict

from qbindiff.features.extractor import FunctionFeatureExtractor, FeatureCollector
from qbindiff.loader import Program, Function


class ChildNb(FunctionFeatureExtractor):
    """
    Function children number.
    This feature extracts the number of functions called by the current one (in call graph).
    """

    key = "cnb"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        value = len(function.children)
        collector.add_feature(self.key, value)


class ParentNb(FunctionFeatureExtractor):
    """
    Function parent number.
    This feature extracts the number of functions calling the current one (in call graph).
    """

    key = "pnb"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        value = len(function.parents)
        collector.add_feature(self.key, value)


class RelativeNb(FunctionFeatureExtractor):
    """
    Function relatives number
    This feature counts both the number of parents and children of the current one (in call graph).
    """

    key = "rnb"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        value = len(function.parents) + len(function.children)
        collector.add_feature(self.key, value)


class LibName(FunctionFeatureExtractor):
    """
    Library (internal) calls feature.
    This features computes a dictionary of library functions called as keys and the count as values.
    It relies on the backend loader to correctly identify a function as a library.
    """

    key = "lib"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        value = defaultdict(int)
        for addr in function.children:
            if program[addr].is_library():
                value[program[addr].name] += 1
        collector.add_dict_feature(self.key, value)


class ImpName(FunctionFeatureExtractor):
    """
    External calls feature.
    It computes a dictionary of external functions called as keys and the count as values.
    External functions are functions imported dynamically.
    """

    key = "imp"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        value = defaultdict(int)
        for addr in function.children:
            if program[addr].is_import():
                value[program[addr].name] += 1
        collector.add_dict_feature(self.key, value)
