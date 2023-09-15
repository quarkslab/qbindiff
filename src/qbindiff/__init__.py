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

"""Quarkslab binary differ

QBinDiff is an experimental binary diffing addressing the diffing as a Network
Alignement Quadratic Problem. Unlike other differs, in QBinDiff you have
control over the diffing process with a lot of tunable parameters.
Additionally QBinDiff can diff any two objects given they implement the
GenericGraph interface.

The key idea is to enable programing the diffing by:

  - writing its own feature
  - being able to enforce some matches
  - being able to put the emphasis on either the content of functions
        (similarity) or the links between them (callgraph)

In essence, the idea is to be able to diff by defining its own criteria which
sometimes, are not the control-flow CFG and instruction but more data-oriented
for instance.

Last, QBinDiff as primarly been designed with the binary-diffing use-case in
mind, but it can be applied to various other use-cases like social-networks.
Indeed, diffing two programs boils down to determining the best alignement of
the call graph following some similarity criterias.

Indeed, solving this problem, APX-hard, that why we use a machine learning
approach (more precisely optimization) to approximate the best match.

Likewise Bindiff, QBinDiff also works using an exported disassembly of program
obtained from a binary analysis platform like IDA, Ghidra or Binary Ninja.
Originally using BinExport, it now also support Quokka as backend that is more
exhaustive on the amount of data exported, faster and more compact on disk
(good for large binary dataset).

Note: QBinDiff is an experimental tool for power-user where many parameters,
thresholds or weights can be adjusted. Use it at your own risks.
"""

from qbindiff.version import __version__
from qbindiff.abstract import GenericGraph
from qbindiff.differ import QBinDiff, DiGraphDiffer, GraphDiffer, Differ
from qbindiff.mapping import Mapping
from qbindiff.loader import Program, Function
from qbindiff.loader.types import LoaderType
from qbindiff.matcher import Matcher
from qbindiff.types import Distance
