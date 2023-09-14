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

"""Features that uses the topology of the CFG
"""

import networkx
import numpy as np
import math
from qbindiff.features.extractor import (
    FunctionFeatureExtractor,
    InstructionFeatureExtractor,
    OperandFeatureExtractor,
    FeatureCollector,
)
from qbindiff.loader import Program, Function, Instruction, Operand
from qbindiff.loader.types import OperandType
import hashlib
import community


class BBlockNb(FunctionFeatureExtractor):
    """
    Number of basic blocks in the function as a feature.
    """

    key = "bnb"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        value = len(function.flowgraph.nodes)
        collector.add_feature(self.key, value)


class StronglyConnectedComponents(FunctionFeatureExtractor):
    """
    Number of strongly connected components in a function CFG.
    """

    key = "scc"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        value = len(list(networkx.strongly_connected_components(function.flowgraph)))
        collector.add_feature(self.key, value)


class BytesHash(FunctionFeatureExtractor):
    """
    Hash of the function, using the instructions sorted by addresses.
    The hashing function used is MD5.
    """

    key = "bh"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        bytes_seq = b""
        for bba, bb in function.items():
            bytes_seq += bb.bytes
        value = float(int(hashlib.md5(bytes_seq).hexdigest(), 16))

        collector.add_feature(self.key, value)


class CyclomaticComplexity(FunctionFeatureExtractor):
    """
    Cyclomatic complexity of the function CFG.
    """

    key = "cc"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        e = len(function.edges)
        n = len([n for n in function.flowgraph.nodes()])
        components = len([c for c in networkx.weakly_connected_components(function.flowgraph)])
        value = e - n + 2 * components
        collector.add_feature(self.key, value)


class MDIndex(FunctionFeatureExtractor):
    """
    MD-Index of the function,
    based on `<https://www.sto.nato.int/publications/STO%20Meeting%20Proceedings/RTO-MP-IST-091/MP-IST-091-26.pdf>`_.
    A slightly modified version of it : notice the topological sort is only available for DAG graphs
    (which may not always be the case)
    """

    key = "mdidx"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        try:
            topological_sort = list(networkx.topological_sort(function.flowgraph))
            sort_ok = True
        except networkx.NetworkXUnfeasible:
            sort_ok = False

        if sort_ok:
            value = np.sum(
                [
                    1
                    / math.sqrt(
                        topological_sort.index(src)
                        + math.sqrt(2) * function.flowgraph.in_degree(src)
                        + math.sqrt(3) * function.flowgraph.out_degree(src)
                        + math.sqrt(5) * function.flowgraph.in_degree(dst)
                        + math.sqrt(7) * function.flowgraph.out_degree(dst)
                    )
                    for (src, dst) in function.edges
                ]
            )
        else:
            value = np.sum(
                [
                    1
                    / math.sqrt(
                        math.sqrt(2) * function.flowgraph.in_degree(src)
                        + math.sqrt(3) * function.flowgraph.out_degree(src)
                        + math.sqrt(5) * function.flowgraph.in_degree(dst)
                        + math.sqrt(7) * function.flowgraph.out_degree(dst)
                    )
                    for (src, dst) in function.edges
                ]
            )

        collector.add_feature(self.key, float(value))


class JumpNb(InstructionFeatureExtractor):
    """
    Number of jumps in the function.
    """

    key = "jnb"

    def visit_instruction(
        self, program: Program, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        if instruction.mnemonic == "jmp":
            collector.add_dict_feature(self.key, {"value": 1})


class SmallPrimeNumbers(FunctionFeatureExtractor):
    """
    Small-Prime-Number based on mnemonics, as defined
    in `Bindiff <https://www.sto.nato.int/publications/STO%20Meeting%20Proceedings/RTO-MP-IST-091/MP-IST-091-26.pdf>`_.
    This hash is slightly different from the theoretical implementation. Modulo is made at each round, instead
    of doing it at the end.
    """

    key = "spp"

    @staticmethod
    def primesbelow(n: int) -> list[int]:
        """
        Utility function that returns a list of all the primes below n.
        This comes from `Diaphora <https://github.com/joxeankoret/diaphora/blob/master/jkutils/factor.py>`_

        :param n: integer n
        :return: list of prime integers below n
        """

        correction = n % 6 > 1
        n = {0: n, 1: n - 1, 2: n + 4, 3: n + 3, 4: n + 2, 5: n + 1}[n % 6]
        sieve = [True] * (n // 3)
        sieve[0] = False
        for i in range(int(n**0.5) // 3 + 1):
            if sieve[i]:
                k = (3 * i + 1) | 1
                sieve[k * k // 3 :: 2 * k] = [False] * ((n // 6 - (k * k) // 6 - 1) // k + 1)
                sieve[(k * k + 4 * k - 2 * k * (i % 2)) // 3 :: 2 * k] = [False] * (
                    (n // 6 - (k * k + 4 * k - 2 * k * (i % 2)) // 6 - 1) // k + 1
                )
        return [2, 3] + [(3 * i + 1) | 1 for i in range(1, n // 3 - correction) if sieve[i]]

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        mnemonics = set()
        for bb_addr, bb in function.items():
            for ins in bb.instructions:
                if ins.mnemonic not in mnemonics:
                    mnemonics.update({ins.mnemonic})

        mnemonics = list(mnemonics)

        value = 1
        primes = self.primesbelow(4096)
        for bb_addr, bb in function.items():
            for ins in bb.instructions:
                value *= primes[mnemonics.index(ins.mnemonic)]
                value = value % (2**64)
        collector.add_feature(self.key, value)


class ReadWriteAccess(OperandFeatureExtractor):
    """
    Number of memory access in the function.
    Both read and write.
    """

    key = "rwa"

    def visit_operand(
        self, program: Program, operand: Operand, collector: FeatureCollector
    ) -> None:
        if operand.type in (OperandType.memory, OperandType.displacement, OperandType.phrase):
            collector.add_dict_feature(self.key, {"value": 1})


class MaxParentNb(FunctionFeatureExtractor):
    """
    Maximum number of parent of a basic block in the function.
    """

    key = "maxp"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        try:
            value = max(
                len(list(function.flowgraph.predecessors(bblock))) for bblock in function.flowgraph
            )
        except ValueError:
            value = 0
        collector.add_feature(self.key, value)


class MaxChildNb(FunctionFeatureExtractor):
    """
    Maximum number of children of a basic block in the function.
    """

    key = "maxc"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        try:
            value = max(
                len(list(function.flowgraph.successors(bblock))) for bblock in function.flowgraph
            )
        except ValueError:
            value = 0
        collector.add_feature(self.key, value)


class MaxInsNB(FunctionFeatureExtractor):
    """
    Max number of instructions per basic blocks in the function.
    """

    key = "maxins"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        try:
            value = max(len(bblock.instructions) for bblock in function)
        except ValueError:
            value = 0
        collector.add_feature(self.key, value)


class MeanInsNB(FunctionFeatureExtractor):
    """
    Mean number of instructions per basic blocks in the function.
    """

    key = "meanins"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        try:
            value = sum(len(bblock.instructions) for bblock in function) / len(function)
        except ValueError:
            value = 0
        collector.add_feature(self.key, value)


class InstNB(FunctionFeatureExtractor):
    """
    Number of instructions in the function.
    """

    key = "totins"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        value = sum(len(bblock.instructions) for bblock in function)
        collector.add_feature(self.key, value)


class GraphMeanDegree(FunctionFeatureExtractor):
    """
    Mean degree of the function.
    """

    key = "Gmd"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        n_node = len(function.flowgraph)
        value = sum(d for _, d in function.flowgraph.degree) / n_node if n_node != 0 else 0
        collector.add_feature(self.key, value)


class GraphDensity(FunctionFeatureExtractor):
    """
    Density of the function flow graph (CFG).
    """

    key = "Gd"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        value = networkx.density(function.flowgraph)
        collector.add_feature(self.key, value)


class GraphNbComponents(FunctionFeatureExtractor):
    """
    Number of components in the function (non-connected flow graphs).
    (This can happen in the way IDA disassemble functions)
    """

    key = "Gnc"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        value = len(list(networkx.connected_components(function.flowgraph.to_undirected())))
        collector.add_feature(self.key, value)


class GraphDiameter(FunctionFeatureExtractor):
    """
    Graph diameter of the function flow graph (CFG).
    """

    key = "Gdi"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        components = list(networkx.connected_components(function.flowgraph.to_undirected()))

        if components:
            value = max(
                networkx.diameter(networkx.subgraph(function.flowgraph, x).to_undirected())
                for x in components
            )
        else:
            value = 0
        collector.add_feature(self.key, value)


class GraphTransitivity(FunctionFeatureExtractor):
    """
    Transitivity of the function flow graph (CFG).
    """

    key = "Gt"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        value = networkx.transitivity(function.flowgraph)
        collector.add_feature(self.key, value)


class GraphCommunities(FunctionFeatureExtractor):
    """
    Number of graph communities (Louvain modularity).
    """

    key = "Gcom"

    def visit_function(
        self, program: Program, function: Function, collector: FeatureCollector
    ) -> None:
        partition = community.best_partition(function.flowgraph.to_undirected())
        if (len(function) > 1) and partition:
            p_list = [x for x in partition.values() if x != function.addr]
            value = max(p_list) if p_list else 0
        else:
            value = 0
        collector.add_feature(self.key, value)
