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

"""Program
"""

from __future__ import annotations
import networkx
from collections.abc import Callable, Iterator

from qbindiff.abstract import GenericGraph
from qbindiff.loader import Function, Structure
from qbindiff.loader.types import LoaderType
from qbindiff.types import Addr
from qbindiff.loader.backend.abstract import AbstractProgramBackend


class Program(dict, GenericGraph):
    """
    Program class that shadows the underlying program backend used.

    It inherits from dict which keys are function addresses and
    values are Function object.

    The node label is the function address, the node itself is the Function object
    """

    def __init__(self, loader: LoaderType | None, /, *args, **kwargs):
        super(Program, self).__init__()
        self._backend = None

        if loader is None and (backend := kwargs.get("backend")) is not None:
            self._backend = backend  # Load directly from instanciated backend

        elif loader == LoaderType.ida:
            from qbindiff.loader.backend.ida import ProgramBackendIDA

            self._backend = ProgramBackendIDA(*args, **kwargs)

        elif loader == LoaderType.binexport:
            from qbindiff.loader.backend.binexport import ProgramBackendBinExport

            self._backend = ProgramBackendBinExport(*args, **kwargs)

        elif loader == LoaderType.quokka:
            from qbindiff.loader.backend.quokka import ProgramBackendQuokka

            self._backend = ProgramBackendQuokka(*args, **kwargs)

        else:
            raise NotImplementedError("Loader: %s not implemented" % loader)

        self._filter = lambda x: True
        self._load_functions()

    @staticmethod
    def from_binexport(file_path: str, enable_cortexm: bool = False) -> Program:
        """
        Load the Program using the binexport backend

        :param file_path: File path to the binexport file
        :param enable_cortexm: Whether to check for cortexm instructions while
                               disassembling with capstone
        :return: Program instance
        """

        return Program(LoaderType.binexport, file_path, enable_cortexm=enable_cortexm)

    @staticmethod
    def from_quokka(file_path: str, exec_path: str) -> Program:
        """
        Load the Program using the Quokka backend.

        :param file_path: File path to the binexport file
        :param exec_path: Path of the raw binary
        :return: Program instance
        """

        return Program(LoaderType.quokka, file_path, exec_path=exec_path)

    @staticmethod
    def from_ida() -> Program:
        """
        Load the program using the IDA backend

        :return: Program instance
        """

        return Program(LoaderType.ida)

    @staticmethod
    def from_backend(backend: AbstractProgramBackend) -> Program:
        """
        Load the Program from an instanciated program backend object
        """

        return Program(None, backend=backend)

    def __repr__(self) -> str:
        return "<Program:%s>" % self.name

    def __iter__(self):
        """
        Override the built-in __iter__ to iterate all functions
        located in the program.

        :return: Iterator of all functions (sorted by address)
        """

        for addr in sorted(self.keys()):
            if self._filter(addr):  # yield function only if filter agree to keep it
                yield self[addr]

    def _load_functions(self) -> None:
        """
        Load the functions from the backend

        :return: None
        """

        for function in map(Function.from_backend, self._backend.functions):
            self[function.addr] = function

    def items(self) -> Iterator[tuple[Addr, Function]]:
        """
        Return an iterator over the items. Each item is {node_label: node}

        :return: an iterator over the program elements. Each element is a tuple of shape (function_addr, function_obj)
        """

        for addr in self.keys():
            if self._filter(addr):  # yield function only if filter agree to keep it
                yield (addr, self[addr])

    def get_node(self, node_label: Addr) -> Function:
        """
        Returns the node identified by the `node_label`

        :param node_label: the node_label or the address from which we want to recover the object
        :return: the function identified by its address
        """

        return self[node_label]

    @property
    def node_labels(self) -> Iterator[Addr]:
        """
        Iterator over the node labels
        """

        for addr in self.keys():
            if self._filter(addr):
                yield addr

    @property
    def nodes(self) -> Iterator[Function]:
        """
        Iterator over the nodes
        """

        yield from self.__iter__()

    @property
    def edges(self) -> Iterator[tuple[Addr, Addr]]:
        """
        Iterator over the edges.
        An edge is a pair (addr_a, addr_b)
        """

        return self.callgraph.edges

    @property
    def name(self) -> str:
        """
        Returns the name of the program as defined by the backend
        """

        return self._backend.name

    @property
    def structures(self) -> list[Structure]:
        """
        Returns the list of structures defined in program
        """

        return self._backend.structures

    @property
    def exec_path(self) -> str | None:
        """
        The executable path if it has been specified, None otherwise
        """

        return self._backend.exec_path

    def set_function_filter(self, func: Callable[[Addr], bool]) -> None:
        """
        Filter out some functions, to ignore them in later processing.

        .. warning: The filter only apply for __iter__, items functions and callgraph
                    property.
                    Accessing functions through the dictionary does not apply the filter

        :param func: function take the function address (the node label) and returns
                     whether or not to keep it.
        """

        self._filter = func

    @property
    def callgraph(self) -> networkx.DiGraph:
        """
        The function callgraph with a Networkx DiGraph
        """

        cg = self._backend.callgraph
        funcs = list(self)  # functions already filtered
        return cg.subgraph([x.addr for x in funcs])

    def get_function(self, name: str) -> Function:
        """
        Returns the function by its name

        :param name: name of the function
        :return: the function
        """

        return self[self._backend.fun_names[name]]

    def follow_through(self, to_remove: Addr, target: Addr) -> None:
        """
        Replace node `to_remove` with a follow-through edge from every parent of the
        node with the node `target`.

        Example : ``{ parents } -> (to_remove) -> (target)``

        ``--> { parents } -> (target)``

        :param to_remove: node to remove
        :param target: targe node
        :return: None
        """

        func = self[to_remove]
        self.pop(to_remove)
        for p_addr in list(func.parents):
            # Remove edges
            self[p_addr].children.remove(to_remove)
            func.parents.remove(p_addr)
            self._backend.callgraph.remove_edge(p_addr, to_remove)
            # Add follow-through edge
            self[p_addr].children.add(target)
            self[target].parents.add(p_addr)
            self._backend.callgraph.add_edge(p_addr, target)
        for c_addr in list(func.children):
            # Remove edges
            func.children.remove(c_addr)
            self[c_addr].parents.remove(to_remove)
            self._backend.callgraph.remove_edge(to_remove, c_addr)
        self._backend.callgraph.remove_node(to_remove)

    def remove_function(self, to_remove: Addr) -> None:
        """
        Remove the node ``to_remove`` from the Call Graph of the program.

        **WARNING**: The follow-through edges from the parents to the children are not
        added. Example :

        ``{ parents } -> (to_remove) -> { children }``

        ``--> { parents }                   { children }``

        :param to_remove: function_to_remove
        :return: None
        """

        func = self[to_remove]
        self.pop(to_remove)
        for p_addr in list(func.parents):
            # Remove edges
            self[p_addr].children.remove(to_remove)
            func.parents.remove(p_addr)
            self._backend.callgraph.remove_edge(p_addr, to_remove)
        for c_addr in list(func.children):
            # Remove edges
            func.children.remove(c_addr)
            self[c_addr].parents.remove(to_remove)
            self._backend.callgraph.remove_edge(to_remove, c_addr)
        self._backend.callgraph.remove_node(to_remove)
